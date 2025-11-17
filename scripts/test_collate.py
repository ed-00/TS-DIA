import numpy as np
import sys
import os
import torch

# Ensure the workspace root is on sys.path so relative package imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.ego_dataset import EgoCentricDiarizationDataset


def create_sample_item(time_frames, feat_dim, speaker_id):
    features = torch.randn(time_frames, feat_dim)
    labels = torch.randint(0, 5, size=(time_frames,), dtype=torch.int64)
    is_target = 0 if speaker_id is None else 1
    return {"features": features, "labels": labels, "speaker_id": speaker_id, "is_target": is_target}


if __name__ == "__main__":
    # Create a synthetic batch with variable lengths and string speaker ids
    batch = [
        create_sample_item(time_frames=100, feat_dim=23, speaker_id="spk1"),
        create_sample_item(time_frames=80, feat_dim=23, speaker_id="spk2"),
        create_sample_item(time_frames=50, feat_dim=23, speaker_id=None),
    ]

    collated = EgoCentricDiarizationDataset.collate_fn(batch)

    # Basic assertions and prints
    assert "features" in collated and "labels" in collated and "is_target" in collated
    assert isinstance(collated["features"], torch.Tensor)
    assert isinstance(collated["labels"], torch.Tensor)
    assert isinstance(collated["is_target"], torch.Tensor)
    assert collated["is_target"].dtype == torch.int64
    # Labels should contain ego classes in range [0, 4] for non-padded positions (ignore pad value)
    IGNORE_INDEX = -100  # from EgoCentricDiarizationDataset.IGNORE_INDEX default
    labels_np = collated["labels"].numpy()
    valid_mask = labels_np != IGNORE_INDEX
    unique_vals = set(labels_np[valid_mask].ravel().tolist()) if valid_mask.any() else set()
    assert all((0 <= v <= 4 for v in unique_vals))

    print("Collate function passed basic smoke tests.")
    print("Feature shape:", collated["features"].shape)
    print("Labels shape:", collated["labels"].shape)
    print("is_target:", collated["is_target"])