#!/usr/bin/env python3
"""
Debug why features are not being processed in the training loop
"""

import torch
from lhotse import CutSet

def debug_dataset_call():
    """Debug how the dataset __getitem__ is being called."""
    print("Creating a mock dataset to test __getitem__ behavior...")
    
    class DebugDataset:
        def __init__(self):
            self.call_count = 0
        
        def __len__(self):
            return 10
        
        def __getitem__(self, item):
            self.call_count += 1
            print(f"Call {self.call_count}: __getitem__ called with: {type(item)} -> {item}")
            
            # Return mock data matching expected format
            return {
                "features": torch.randn(4, 100, 345),  # Processed features
                "features_lens": torch.tensor([100, 95, 98, 92]),
                "enroll_features": torch.randn(4, 50, 345),
                "enroll_features_lens": torch.tensor([50, 48, 45, 49]),
                "labels": torch.randint(0, 5, (4, 100))
            }
    
    dataset = DebugDataset()
    
    # Test 1: Direct index access (PyTorch DataLoader style)
    print("\nTest 1: Direct index access")
    try:
        result = dataset[0]
        print(f"Success: {result['features'].shape}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 2: CutSet access (Lhotse sampler style)
    print("\nTest 2: CutSet access (Lhotse style)")
    try:
        # Create a mock CutSet
        from lhotse import Recording, AudioSource
        from lhotse.cut import Cut
        
        recording = Recording(
            id="mock_rec",
            sources=[AudioSource(type="file", channels=[0], source="mock.wav")],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0
        )
        
        cut = recording.to_cut().with_id("mock_cut")
        cutset = CutSet.from_cuts([cut])
        
        result = dataset[cutset]
        print(f"Success: {result['features'].shape}")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Test 3: List of indices access
    print("\nTest 3: List of indices access")
    try:
        result = dataset[[0, 1, 2, 3]]
        print(f"Success: {result['features'].shape}")
    except Exception as e:
        print(f"Failed: {e}")

def check_lhotse_dataset_pattern():
    """Check how Lhotse's DiarizationDataset works."""
    print("\n" + "="*50)
    print("Checking Lhotse's DiarizationDataset pattern...")
    
    try:
        from lhotse.dataset import DiarizationDataset
        
        # Create a simple cut for testing
        from lhotse import Recording, AudioSource, SupervisionSegment
        
        recording = Recording(
            id="test",
            sources=[AudioSource(type="file", channels=[0], source="test.wav")],
            sampling_rate=16000,
            num_samples=160000,
            duration=10.0
        )
        
        cut = recording.to_cut().with_id("test_cut")
        cut.supervisions = [
            SupervisionSegment(
                id="seg1", recording_id="test", start=1.0, duration=2.0, 
                channel=0, speaker="spk1"
            )
        ]
        
        cuts = CutSet.from_cuts([cut])
        lhotse_dataset = DiarizationDataset(cuts)
        
        print(f"Lhotse DiarizationDataset length: {len(lhotse_dataset)}")
        print("Lhotse DiarizationDataset __getitem__ signature:")
        import inspect
        sig = inspect.signature(lhotse_dataset.__getitem__)
        print(f"  {sig}")
        
        # Try calling it with a CutSet
        try:
            batch_cuts = CutSet.from_cuts([cut])
            result = lhotse_dataset[batch_cuts]
            print(f"Lhotse dataset result keys: {result.keys()}")
            if 'features' in result:
                print(f"Lhotse features shape: {result['features'].shape}")
            
        except Exception as e:
            print(f"Lhotse dataset call failed: {e}")
            
    except Exception as e:
        print(f"Could not test Lhotse dataset: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Debugging Dataset Call Patterns")
    print("=" * 60)
    
    debug_dataset_call()
    check_lhotse_dataset_pattern()
    
    print("\n" + "=" * 60)
    print("The issue is likely that our EgoCentricDiarizationDataset")
    print("needs to follow Lhotse's exact calling pattern.")
    print("=" * 60)