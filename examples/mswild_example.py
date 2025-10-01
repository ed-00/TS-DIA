#!/usr/bin/env python3
"""
Example script demonstrating how to download and prepare the MSDWILD dataset.

This script shows how to use the MSDWILD recipe functions to download and process
the multi-modal speaker diarization dataset.
"""

import logging
from pathlib import Path

from datasets.recipes.mswild import download_mswild, prepare_mswild

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating MSDWILD dataset usage."""

    # Set up paths
    data_dir = Path("./data/mswild")
    output_dir = Path("./manifests/mswild")

    logger.info("Starting MSDWILD dataset download and preparation...")

    # Download the dataset
    logger.info("Downloading MSDWILD dataset...")
    download_mswild(
        target_dir=data_dir,
        force_download=False,  # Set to True to re-download
        download_audio=True,  # Download audio files (required)
        download_video=False,  # Skip video files (large ~43GB)
        download_faces=False,  # Skip cropped faces (large ~14GB)
    )

    # Prepare the dataset
    logger.info("Preparing MSDWILD dataset...")
    manifests = prepare_mswild(
        corpus_dir=data_dir,
        output_dir=output_dir,
        splits={"train": "few_train", "dev": "few_val", "test": "many_val"},
    )

    # Print summary
    logger.info("Dataset preparation completed!")
    for split_name, manifest in manifests.items():
        recordings = manifest["recordings"]
        supervisions = manifest["supervisions"]
        logger.info(
            f"{split_name}: {len(recordings)} recordings, {len(supervisions)} supervision segments"
        )

        # Print some statistics
        if len(recordings) > 0:
            total_duration = sum(rec.duration for rec in recordings)
            logger.info(f"  Total duration: {total_duration:.2f} seconds")

        if len(supervisions) > 0:
            unique_speakers = len(
                set(sup.speaker for sup in supervisions if sup.speaker)
            )
            logger.info(f"  Unique speakers: {unique_speakers}")


if __name__ == "__main__":
    main()
