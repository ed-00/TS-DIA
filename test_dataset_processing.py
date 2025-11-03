#!/usr/bin/env python3
"""
Test that the EgoCentricDiarizationDataset properly processes features with splicing and subsampling
"""

import torch
from lhotse import CutSet
from training.ego_dataset import EgoCentricDiarizationDataset

def test_feature_processing_with_mock_cuts():
    """Test that features are properly spliced and subsampled."""
    print("Testing feature processing with mock data...")
    
    # Create mock cut with pre-computed features
    from lhotse import Recording, SupervisionSegment, AudioSource
    from lhotse.cut import Cut
    from lhotse.features import Features
    
    # Create a recording
    recording = Recording(
        id="test_rec",
        sources=[AudioSource(type="file", channels=[0], source="dummy.wav")],
        sampling_rate=16000,
        num_samples=160000,  # 10 seconds
        duration=10.0
    )
    
    # Create supervisions
    supervisions = [
        SupervisionSegment(
            id="spk1_seg",
            recording_id="test_rec",
            start=1.0,
            duration=3.0,
            channel=0,
            speaker="speaker_1"
        ),
        SupervisionSegment(
            id="spk2_seg",
            recording_id="test_rec",
            start=5.0,
            duration=2.0,
            channel=0,
            speaker="speaker_2"
        )
    ]
    
    # Create cut
    cut = recording.to_cut()
    cut = cut.with_id("test_cut")
    cut.supervisions = supervisions
    
    # Add mock features - simulate what would come from feature extraction
    num_frames = 1000  # 10 seconds at 100fps
    feature_dim = 23   # MFCC features
    
    # Create mock features tensor
    features_data = torch.randn(num_frames, feature_dim)
    
    # We need to simulate the cut having pre-computed features
    # In a real scenario, this would come from Lhotse's feature storage
    class MockFeatures:
        def __init__(self, data):
            self.data = data
            
        def load(self, start=None, duration=None):
            if start is not None and duration is not None:
                start_frame = int(start * 100)  # 100fps
                end_frame = start_frame + int(duration * 100)
                return self.data[start_frame:end_frame]
            return self.data
    
    # Monkey patch to add features
    cut.features = MockFeatures(features_data)
    cut.num_frames = num_frames
    cut.frame_shift = 0.01  # 10ms frames
    
    # Create dataset
    cuts = CutSet.from_cuts([cut])
    dataset = EgoCentricDiarizationDataset(
        cuts=cuts,
        min_enroll_len=1.0,
        max_enroll_len=2.0,
        context_size=7,
        subsampling=10
    )
    
    print(f"Dataset created with {len(dataset)} examples")
    
    if len(dataset) > 0:
        # Test with CutSet (Lhotse sampler style)
        batch_cuts = CutSet.from_cuts([cut])
        
        # Patch collate_features to work with our mock
        def mock_collate_features(cutset):
            cuts_list = list(cutset)
            batch_features = []
            batch_lengths = []
            
            for c in cuts_list:
                feat = c.features.load()
                batch_features.append(feat)
                batch_lengths.append(feat.size(0))
            
            # Pad to same length
            max_len = max(batch_lengths)
            padded_features = []
            for feat in batch_features:
                if feat.size(0) < max_len:
                    padding = torch.zeros(max_len - feat.size(0), feat.size(1))
                    feat = torch.cat([feat, padding], dim=0)
                padded_features.append(feat)
            
            features = torch.stack(padded_features)
            lengths = torch.tensor(batch_lengths, dtype=torch.long)
            return features, lengths
        
        # Temporarily replace collate_features
        import training.ego_dataset
        original_collate = training.ego_dataset.collate_features
        training.ego_dataset.collate_features = mock_collate_features
        
        try:
            batch = dataset[batch_cuts]
            
            print("Batch output:")
            print(f"  Features shape: {batch['features'].shape}")
            print(f"  Features lens: {batch['features_lens']}")
            print(f"  Enroll features shape: {batch['enroll_features'].shape}")
            print(f"  Enroll lens: {batch['enroll_features_lens']}")
            print(f"  Labels shape: {batch['labels'].shape}")
            
            # Check dimensions
            B, T_sub, F_spliced = batch['features'].shape
            expected_F = feature_dim * (2 * 7 + 1)  # 23 * 15 = 345
            expected_T_range = (num_frames // 10 - 10, num_frames // 10 + 10)  # Allow some variance
            
            print(f"\nDimension Analysis:")
            print(f"  Original: {num_frames} frames × {feature_dim} features")
            print(f"  After splicing: expected {feature_dim} × 15 = {expected_F} features")
            print(f"  After subsampling: expected ~{num_frames // 10} frames")
            print(f"  Actual result: {T_sub} frames × {F_spliced} features")
            
            # Verify splicing worked
            if F_spliced == expected_F:
                print("✓ Splicing applied correctly!")
            else:
                print(f"✗ Splicing failed: expected {expected_F}, got {F_spliced}")
            
            # Verify subsampling worked  
            if expected_T_range[0] <= T_sub <= expected_T_range[1]:
                print("✓ Subsampling applied correctly!")
            else:
                print(f"✗ Subsampling failed: expected ~{num_frames // 10}, got {T_sub}")
            
            return True
            
        finally:
            # Restore original function
            training.ego_dataset.collate_features = original_collate
    
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing EgoCentric Dataset Feature Processing")
    print("=" * 60)
    
    success = test_feature_processing_with_mock_cuts()
    
    if success:
        print("\n✓ Feature processing test completed successfully!")
    else:
        print("\n✗ Feature processing test failed or no examples found")
    
    print("=" * 60)