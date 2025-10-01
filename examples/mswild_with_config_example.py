#!/usr/bin/env python3
"""
Example script demonstrating how to use MSDWILD dataset with the TS-DIA dataset management system.

This script shows how to:
1. Use YAML configuration files to manage MSDWILD dataset
2. Parse configurations and apply global defaults
3. Use the dataset management system for downloading and processing
"""

import logging
from pathlib import Path

from datasets.parse_args import parse_dataset_configs
from datasets.recipes.mswild import download_mswild, prepare_mswild

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating MSDWILD dataset usage with configuration management."""

    # Path to the configuration file
    config_path = Path(__file__).parent / "mswild_config.yml"

    logger.info("Starting MSDWILD dataset management with configuration...")

    try:
        # Parse the configuration file
        dataset_configs = parse_dataset_configs(config_path)

        # Process each dataset configuration
        for config in dataset_configs:
            logger.info(f"Processing dataset: {config.name}")

            # Get parameters
            download_kwargs = config.get_download_kwargs()
            process_kwargs = config.get_process_kwargs()

            logger.info(f"Download parameters: {download_kwargs}")
            logger.info(f"Process parameters: {process_kwargs}")

            # Download the dataset
            logger.info("Downloading MSDWILD dataset...")
            download_mswild(**download_kwargs)

            # Prepare the dataset
            logger.info("Preparing MSDWILD dataset...")
            manifests = prepare_mswild(**process_kwargs)

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

    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise


def demonstrate_config_parsing():
    """Demonstrate how configuration parsing works."""

    logger.info("Demonstrating configuration parsing...")

    # Example of parsing a configuration
    config_path = Path(__file__).parent / "mswild_config.yml"

    try:
        dataset_configs = parse_dataset_configs(config_path)

        for i, config in enumerate(dataset_configs):
            logger.info(f"\nDataset {i + 1}: {config.name}")
            logger.info(f"  Download params: {config.download_params}")
            logger.info(f"  Process params: {config.process_params}")

            # Show how global config is applied
            logger.info(f"  Download kwargs: {config.get_download_kwargs()}")
            logger.info(f"  Process kwargs: {config.get_process_kwargs()}")

    except Exception as e:
        logger.error(f"Error parsing configuration: {e}")
        raise


if __name__ == "__main__":
    # Run the main example
    main()

    # Also demonstrate configuration parsing
    print("\n" + "=" * 50)
    demonstrate_config_parsing()
