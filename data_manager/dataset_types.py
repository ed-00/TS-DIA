#! /usr/bin/env python
# copyright (c) 2025 Abed Hameed.
# Licensed under Apache 2.0 license.
"""
Dataset Types and Configuration Classes

This module defines the core data structures and configuration classes for the TS-DIA
dataset management system. It provides a hybrid approach supporting both typed dataclasses
and dictionary-based configurations with global defaults.

Key Classes:
    GlobalConfig: Global configuration defaults for all datasets
    DatasetConfig: Individual dataset configuration with global config merging
    LoadDatasetsParams: Parameters for loading multiple datasets
    BaseDownloadParams: Base class for dataset download parameters
    BaseProcessParams: Base class for dataset processing parameters

The module also defines specific parameter classes for 50+ supported datasets,
including TIMIT, LibriSpeech, VoxCeleb, AMI, and many others.

Example:
    ```python
    from datasets.dataset_types import DatasetConfig, GlobalConfig

    # Create global configuration
    global_config = GlobalConfig(
        corpus_dir="./data",
        output_dir="./manifests",
        force_download=False
    )

    # Create dataset configuration
    dataset_config = DatasetConfig(
        name="timit",
        process_params={"num_phones": 48}
    )

    # Apply global config
    dataset_config.apply_global_config(global_config)
    ```
"""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from lhotse.utils import Pathlike


@dataclass
class BaseDownloadParams(ABC):
    """Base class for download parameters with common fields across all datasets"""

    target_dir: Pathlike = "."
    force_download: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class BaseProcessParams(ABC):
    """Base class for process parameters with common fields across all datasets"""

    output_dir: Optional[Pathlike] = None
    corpus_dir: Optional[Pathlike] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ============================================================================
# Feature Configuration
# ============================================================================


@dataclass
class FeatureConfig:
    """
    Feature extraction configuration for audio processing.

    This dataclass defines all parameters for extracting acoustic features
    from audio data using Lhotse. It supports multiple feature types (fbank, mfcc, spectrogram)
    and provides comprehensive control over feature extraction.

    Attributes:
        feature_type: Type of features to extract ('fbank', 'mfcc', 'spectrogram')

        # Basic parameters
        sampling_rate: Target sampling rate in Hz
        frame_length: Frame/window length in seconds (default: 25ms)
        frame_shift: Frame shift/hop length in seconds (default: 10ms)

        # Window and FFT parameters
        round_to_power_of_two: Round window size to power of 2
        remove_dc_offset: Remove DC offset before processing
        preemph_coeff: Pre-emphasis coefficient (typically 0.97)
        window_type: Window function type ('povey', 'hanning', 'hamming', etc.)
        dither: Dithering factor for numerical stability
        snip_edges: Whether to snip edges during feature extraction

        # Energy parameters
        energy_floor: Floor value for energy computation
        raw_energy: Use raw energy (before windowing)
        use_energy: Use energy instead of C0 for MFCC
        use_fft_mag: Use FFT magnitude instead of power

        # Mel filterbank parameters
        low_freq: Lower frequency bound in Hz
        high_freq: Upper frequency bound in Hz (-400 means nyquist - 400)
        num_filters: Number of triangular mel filters (for spectrogram)
        torchaudio_compatible_mel_scale: Use torchaudio-compatible mel scale
        num_mel_bins: Number of mel-frequency bins (for fbank/mfcc)
        norm_filters: Normalize mel filter weights

        # MFCC-specific parameters
        num_ceps: Number of cepstral coefficients (MFCC only)
        cepstral_lifter: Cepstral liftering coefficient (MFCC only)

        # Feature computation and storage parameters
        storage_path: Path to store features (None for in-memory)
        num_jobs: Number of parallel jobs for feature extraction
        torch_threads: PyTorch thread count (None = auto-set to 1 when num_jobs > 1)
        storage_type: Storage format ('lilcom_chunky', 'lilcom_files', 'numpy', 'hdf5')
        mix_eagerly: Whether to mix cuts eagerly during extraction
        progress_bar: Show progress bar during feature extraction

        # Hardware
        device: Device for computation ('cpu', 'cuda')

        # Windowing of long recordings before feature extraction
        cut_window_seconds: If set (>0), pre-window long recordings to fixed-length cuts
    """

    # Feature type
    feature_type: Literal["fbank", "mfcc", "spectrogram"] = "fbank"

    # Basic parameters
    sampling_rate: int = 16000
    frame_length: float = 0.025  # 25ms
    frame_shift: float = 0.01  # 10ms

    # Window and FFT parameters
    round_to_power_of_two: bool = True
    remove_dc_offset: bool = True
    preemph_coeff: float = 0.97
    window_type: str = "povey"
    dither: float = 0.0
    snip_edges: bool = False

    # Energy parameters
    energy_floor: float = 1e-10  # EPSILON
    raw_energy: bool = True
    use_energy: bool = False
    use_fft_mag: bool = False

    # Mel filterbank parameters
    low_freq: float = 20.0
    high_freq: float = -400.0  # -400 means nyquist - 400
    num_filters: int = 23
    torchaudio_compatible_mel_scale: bool = True
    num_mel_bins: Optional[int] = 80
    norm_filters: bool = False

    # MFCC-specific parameters
    num_ceps: int = 13
    cepstral_lifter: int = 22

    # Feature computation and storage parameters
    storage_path: Optional[str] = None  # Path to store features (None = in-memory)
    num_jobs: Optional[int] = 1  # Number of parallel jobs for feature extraction
    torch_threads: Optional[int] = (
        None  # PyTorch thread count (None = auto-set to 1 when num_jobs > 1)
    )
    storage_type: str = "lilcom_chunky"  # Storage type: 'lilcom_chunky', 'lilcom_files', 'numpy', 'hdf5'
    mix_eagerly: bool = True  # Whether to mix cuts eagerly during feature extraction
    progress_bar: bool = True  # Show progress bar during feature extraction

    # Hardware
    device: str = "cpu"

    # Windowing of long recordings before feature extraction
    cut_window_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ============================================================================
# Global Configuration
# ============================================================================


@dataclass
class InputStrategyConfig:
    """Configuration for Lhotse InputStrategies parallelism and behavior."""

    num_workers: int = 0
    executor_type: Literal["thread", "process"] = "thread"
    # The following are primarily relevant for OnTheFlyFeatures
    use_batch_extract: bool = True
    fault_tolerant: bool = False
    return_audio: bool = False


@dataclass
class SamplerConfig:
    """Configuration for sampler selection and batching policy."""

    type: Literal["dynamic_bucketing", "bucketing", "simple"] = "bucketing"
    num_buckets: int = 10
    max_duration: Optional[float] = None
    shuffle: bool = True
    drop_last: bool = True


@dataclass
class DataLoaderConfig:
    """Configuration for torch DataLoader settings."""

    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = 2


@dataclass
class DataLoadingConfig:
    """Top-level data loading configuration driven from YAML.

    Controls input strategy (precomputed vs on-the-fly), sampler selection,
    and DataLoader worker settings.
    """

    strategy: Literal[
        "precomputed_features",
        "on_the_fly_features",
        "audio_samples",
    ] = "precomputed_features"
    input_strategy: InputStrategyConfig = field(default_factory=InputStrategyConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)


@dataclass
class GlobalConfig:
    """
    Global configuration for all datasets.

    This configuration provides default settings that apply to all datasets
    in the pipeline, including data directories and feature extraction parameters.

    Attributes:
        corpus_dir: Directory where raw dataset files are stored
        output_dir: Directory for output manifests
        force_download: Force re-download of datasets
        storage_path: Path to store extracted features (for caching)
        features: Feature extraction configuration
        data_loading: Data pipeline configuration (strategy, sampler, dataloader)
        random_seed: Global seed used when training.random_seed is absent
    """

    corpus_dir: str = "./data"
    output_dir: str = "./manifests"
    force_download: bool = False
    storage_path: str | None = None  # Feature storage path for caching
    features: FeatureConfig = field(default_factory=FeatureConfig)
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls"""
        result = {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and k not in {"features", "data_loading"}
        }
        # Include individual feature fields for backward compatibility
        if self.features:
            result.update(self.features.to_dict())
        # Expose a flattened view of data loading knobs for legacy callers, if needed
        if self.data_loading:
            # Only include simple scalars commonly used; nested structures should be accessed via the object
            result.update(
                {
                    "data_loading_strategy": self.data_loading.strategy,
                    "dataloader_num_workers": self.data_loading.dataloader.num_workers,
                }
            )
        return result

    def get_feature_config(self) -> FeatureConfig:
        """Get feature extraction configuration"""
        return self.features


# ============================================================================
# Dataset Configuration
# ============================================================================


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset with hybrid typed + dict approach"""

    name: str
    download_params: Union[BaseDownloadParams, Dict[str, Any]] = field(
        default_factory=BaseDownloadParams
    )
    process_params: Union[BaseProcessParams, Dict[str, Any]] = field(
        default_factory=BaseProcessParams
    )

    def __post_init__(self):
        """Convert dict parameters to appropriate typed classes if needed"""
        if isinstance(self.download_params, dict):
            # Create a minimal download params instance and update with dict values
            base_params = BaseDownloadParams()
            for key, value in self.download_params.items():
                if hasattr(base_params, key):
                    setattr(base_params, key, value)
            self.download_params = base_params

        if isinstance(self.process_params, dict):
            # Create a minimal process params instance and update with dict values
            base_params = BaseProcessParams()
            for key, value in self.process_params.items():
                if hasattr(base_params, key):
                    setattr(base_params, key, value)
            self.process_params = base_params

    def get_download_kwargs(self) -> Dict[str, Any]:
        """Get download parameters as dictionary"""
        kwargs = self.download_params if isinstance(self.download_params, dict) else self.download_params.to_dict()

        # Ensure target_dir is dataset-specific by appending dataset name
        if "target_dir" in kwargs:
            kwargs["target_dir"] = str(Path(kwargs["target_dir"]) / self.name)

        return kwargs

    def get_process_kwargs(self) -> Dict[str, Any]:
        """Get process parameters as dictionary"""
        if isinstance(self.process_params, dict):
            return self.process_params
        return self.process_params.to_dict()

    def apply_global_config(
        self, global_config: Union[GlobalConfig, Dict[str, Any]]
    ) -> None:
        """Apply global configuration to this dataset config"""
        if isinstance(global_config, dict):
            # Convert dict to GlobalConfig for consistency
            global_config = GlobalConfig(**global_config)

        # Apply global download params
        if isinstance(self.download_params, dict):
            if "force_download" not in self.download_params:
                self.download_params["force_download"] = global_config.force_download
        else:
            if (
                not hasattr(self.download_params, "force_download")
                or self.download_params.force_download is None
            ):
                self.download_params.force_download = global_config.force_download

        # Apply global process params
        if isinstance(self.process_params, dict):
            if "corpus_dir" not in self.process_params:
                self.process_params["corpus_dir"] = (
                    f"{global_config.corpus_dir}/{self.name}"
                )
            if "output_dir" not in self.process_params:
                self.process_params["output_dir"] = global_config.output_dir
        else:
            if (
                not hasattr(self.process_params, "corpus_dir")
                or self.process_params.corpus_dir is None
            ):
                self.process_params.corpus_dir = (
                    f"{global_config.corpus_dir}/{self.name}"
                )
            if (
                not hasattr(self.process_params, "output_dir")
                or self.process_params.output_dir is None
            ):
                self.process_params.output_dir = global_config.output_dir


@dataclass
class LoadDatasetsParams:
    """Parameters for loading datasets"""

    datasets: List[DatasetConfig]
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    validation_split: float = 0.1
    test_split: float = 0.1


# ============================================================================
# ADEPT DATASET
# ============================================================================


@dataclass
class AdeptDownloadParams(BaseDownloadParams):
    """Download parameters for ADEPT dataset"""

    pass


@dataclass
class AdeptProcessParams(BaseProcessParams):
    """Process parameters for ADEPT dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# AISHELL DATASET
# ============================================================================


@dataclass
class AishellDownloadParams(BaseDownloadParams):
    """Download parameters for AISHELL dataset"""

    base_url: str = "http://www.openslr.org/resources"


@dataclass
class AishellProcessParams(BaseProcessParams):
    """Process parameters for AISHELL dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# AISHELL3 DATASET
# ============================================================================


@dataclass
class Aishell3DownloadParams(BaseDownloadParams):
    """Download parameters for AISHELL3 dataset"""

    base_url: Optional[str] = "http://www.openslr.org/resources"


@dataclass
class Aishell3ProcessParams(BaseProcessParams):
    """Process parameters for AISHELL3 dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# AISHELL4 DATASET
# ============================================================================


@dataclass
class Aishell4DownloadParams(BaseDownloadParams):
    """Download parameters for AISHELL4 dataset"""

    base_url: Optional[str] = "http://www.openslr.org/resources"


@dataclass
class Aishell4ProcessParams(BaseProcessParams):
    """Process parameters for AISHELL4 dataset"""

    corpus_dir: Pathlike = None
    normalize_text: bool = False
    sampling_rate: Optional[int] = None


# ============================================================================
# ALI MEETING DATASET
# ============================================================================


@dataclass
class AliMeetingDownloadParams(BaseDownloadParams):
    """Download parameters for ALI MEETING dataset"""

    base_url: Optional[str] = (
        "https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/"
    )


@dataclass
class AliMeetingProcessParams(BaseProcessParams):
    """Process parameters for ALI MEETING dataset"""

    corpus_dir: Pathlike = None
    mic: Optional[str] = "far"
    normalize_text: str = "none"
    save_mono: bool = False


# ============================================================================
# AMI DATASET
# ============================================================================


@dataclass
class AmiDownloadParams(BaseDownloadParams):
    """Download parameters for AMI dataset"""

    annotations: Optional[Pathlike] = None
    url: Optional[str] = "http://groups.inf.ed.ac.uk/ami"
    mic: Optional[str] = "ihm"


@dataclass
class AmiProcessParams(BaseProcessParams):
    """Process parameters for AMI dataset"""

    data_dir: Pathlike = None
    annotations_dir: Optional[Pathlike] = None
    mic: Optional[str] = "ihm"
    partition: Optional[str] = "full-corpus"
    normalize_text: str = "kaldi"
    max_words_per_segment: Optional[int] = None
    merge_consecutive: bool = False
    sampling_rate: Optional[int] = None



# ============================================================================
# ATCOSIM DATASET
# ============================================================================


@dataclass
class AtcosimDownloadParams(BaseDownloadParams):
    """Download parameters for ATCOSIM dataset"""

    pass


@dataclass
class AtcosimProcessParams(BaseProcessParams):
    """Process parameters for ATCOSIM dataset"""

    corpus_dir: Pathlike = None
    silence_sym: Optional[str] = ""
    breath_sym: Optional[str] = ""
    foreign_sym: Optional[str] = "<unk>"
    partial_sym: Optional[str] = "<unk>"
    unknown_sym: Optional[str] = "<unk>"


# ============================================================================
# BAKER ZH DATASET
# ============================================================================


@dataclass
class BakerZhDownloadParams(BaseDownloadParams):
    """Download parameters for BAKER ZH dataset"""

    pass


@dataclass
class BakerZhProcessParams(BaseProcessParams):
    """Process parameters for BAKER ZH dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# BUT REVERB DB DATASET
# ============================================================================


@dataclass
class ButReverbDbDownloadParams(BaseDownloadParams):
    """Download parameters for BUT REVERB DB dataset"""

    url: Optional[str] = None  # BUT_REVERB_DB_URL


@dataclass
class ButReverbDbProcessParams(BaseProcessParams):
    """Process parameters for BUT REVERB DB dataset"""

    corpus_dir: Pathlike = None
    parts: Sequence[str] = ("silence", "rir")


# ============================================================================
# BVCC DATASET
# ============================================================================


@dataclass
class BvccDownloadParams(BaseDownloadParams):
    """Download parameters for BVCC dataset"""

    pass


@dataclass
class BvccProcessParams(BaseProcessParams):
    """Process parameters for BVCC dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# CHIME6 DATASET
# ============================================================================


@dataclass
class Chime6DownloadParams(BaseDownloadParams):
    """Download parameters for CHIME6 dataset"""

    pass


@dataclass
class Chime6ProcessParams(BaseProcessParams):
    """Process parameters for CHIME6 dataset"""

    corpus_dir: Pathlike = None
    dataset_parts: Optional[Union[str, Sequence[str]]] = "all"
    mic: str = "mdm"
    use_reference_array: bool = False
    perform_array_sync: bool = False
    verify_md5_checksums: bool = False
    num_threads_per_job: int = 1
    sox_path: Pathlike = "/usr/bin/sox"
    normalize_text: str = "kaldi"
    use_chime7_split: bool = False


# ============================================================================
# CMU ARCTIC DATASET
# ============================================================================


@dataclass
class CmuArcticDownloadParams(BaseDownloadParams):
    """Download parameters for CMU ARCTIC dataset"""

    speakers: Sequence[str] = None  # SPEAKERS
    base_url: Optional[str] = None  # BASE_URL


@dataclass
class CmuArcticProcessParams(BaseProcessParams):
    """Process parameters for CMU ARCTIC dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# CMU INDIC DATASET
# ============================================================================


@dataclass
class CmuIndicDownloadParams(BaseDownloadParams):
    """Download parameters for CMU INDIC dataset"""

    speakers: Sequence[str] = None  # SPEAKERS
    base_url: Optional[str] = None  # BASE_URL


@dataclass
class CmuIndicProcessParams(BaseProcessParams):
    """Process parameters for CMU INDIC dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# DAILY TALK DATASET
# ============================================================================


@dataclass
class DailyTalkDownloadParams(BaseDownloadParams):
    """Download parameters for DAILY TALK dataset"""

    pass


@dataclass
class DailyTalkProcessParams(BaseProcessParams):
    """Process parameters for DAILY TALK dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# DIPCO DATASET
# ============================================================================


@dataclass
class DipcoDownloadParams(BaseDownloadParams):
    """Download parameters for DIPCO dataset"""

    pass


@dataclass
class DipcoProcessParams(BaseProcessParams):
    """Process parameters for DIPCO dataset"""

    corpus_dir: Pathlike = None
    mic: Optional[str] = "mdm"
    normalize_text: Optional[str] = "kaldi"
    use_chime7_offset: Optional[bool] = False


# ============================================================================
# EARNINGS21 DATASET
# ============================================================================


@dataclass
class Earnings21DownloadParams(BaseDownloadParams):
    """Download parameters for EARNINGS21 dataset"""

    url: Optional[str] = None  # _DEFAULT_URL


@dataclass
class Earnings21ProcessParams(BaseProcessParams):
    """Process parameters for EARNINGS21 dataset"""

    corpus_dir: Pathlike = None
    normalize_text: bool = False


# ============================================================================
# EARNINGS22 DATASET
# ============================================================================


@dataclass
class Earnings22DownloadParams(BaseDownloadParams):
    """Download parameters for EARNINGS22 dataset"""

    url: Optional[str] = None  # _DEFAULT_URL


@dataclass
class Earnings22ProcessParams(BaseProcessParams):
    """Process parameters for EARNINGS22 dataset"""

    corpus_dir: Pathlike = None
    normalize_text: bool = False


# ============================================================================
# EARS DATASET
# ============================================================================


@dataclass
class EarsDownloadParams(BaseDownloadParams):
    """Download parameters for EARS dataset"""

    pass


@dataclass
class EarsProcessParams(BaseProcessParams):
    """Process parameters for EARS dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# EDACC DATASET
# ============================================================================


@dataclass
class EdaccDownloadParams(BaseDownloadParams):
    """Download parameters for EDACC dataset"""

    base_url: str = "https://datashare.ed.ac.uk/download/"


@dataclass
class EdaccProcessParams(BaseProcessParams):
    """Process parameters for EDACC dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# FLEURS DATASET
# ============================================================================


@dataclass
class FleursDownloadParams(BaseDownloadParams):
    """Download parameters for FLEURS dataset"""

    languages: Optional[Union[str, Sequence[str]]] = "all"


@dataclass
class FleursProcessParams(BaseProcessParams):
    """Process parameters for FLEURS dataset"""

    corpus_dir: Pathlike = None
    languages: Optional[Union[str, Sequence[str]]] = "all"


# ============================================================================
# GIGAST DATASET
# ============================================================================


@dataclass
class GigastDownloadParams(BaseDownloadParams):
    """Download parameters for GIGAST dataset"""

    languages: Union[str, Sequence[str]] = "all"


@dataclass
class GigastProcessParams(BaseProcessParams):
    """Process parameters for GIGAST dataset"""

    corpus_dir: Pathlike = None
    manifests_dir: Pathlike = None
    languages: Union[str, Sequence[str]] = "auto"
    dataset_parts: Union[str, Sequence[str]] = "auto"


# ============================================================================
# GRID DATASET
# ============================================================================


@dataclass
class GridDownloadParams(BaseDownloadParams):
    """Download parameters for GRID dataset"""

    pass


@dataclass
class GridProcessParams(BaseProcessParams):
    """Process parameters for GRID dataset"""

    corpus_dir: Pathlike = None
    output_dir: Optional[Pathlike] = None
    with_supervisions: bool = True


# ============================================================================
# HEROICO DATASET
# ============================================================================


@dataclass
class HeroicoDownloadParams(BaseDownloadParams):
    """Download parameters for HEROICO dataset"""

    url: Optional[str] = "http://www.openslr.org/resources/39"


@dataclass
class HeroicoProcessParams(BaseProcessParams):
    """Process parameters for HEROICO dataset"""

    speech_dir: Pathlike = None
    transcript_dir: Pathlike = None


# ============================================================================
# HIFITTS DATASET
# ============================================================================


@dataclass
class HifittsDownloadParams(BaseDownloadParams):
    """Download parameters for HIFITTS dataset"""

    base_url: Optional[str] = "http://www.openslr.org/resources"


@dataclass
class HifittsProcessParams(BaseProcessParams):
    """Process parameters for HIFITTS dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# HIMIA DATASET
# ============================================================================


@dataclass
class HimiaDownloadParams(BaseDownloadParams):
    """Download parameters for HIMIA dataset"""

    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto"
    base_url: str = "http://www.openslr.org/resources"


@dataclass
class HimiaProcessParams(BaseProcessParams):
    """Process parameters for HIMIA dataset"""

    corpus_dir: Pathlike = None
    dataset_parts: Union[str, Sequence[str]] = "auto"


# ============================================================================
# ICSI DATASET
# ============================================================================


@dataclass
class IcsiDownloadParams(BaseDownloadParams):
    """Download parameters for ICSI dataset"""

    audio_dir: Optional[Pathlike] = None
    transcripts_dir: Optional[Pathlike] = None
    url: Optional[str] = "http://groups.inf.ed.ac.uk/ami"
    mic: Optional[str] = "ihm"


@dataclass
class IcsiProcessParams(BaseProcessParams):
    """Process parameters for ICSI dataset"""

    audio_dir: Pathlike = None
    transcripts_dir: Optional[Pathlike] = None
    mic: Optional[str] = "ihm"
    normalize_text: str = "kaldi"
    save_to_wav: bool = False
    sampling_rate: Optional[int] = None

# ============================================================================
# LIBRICSS DATASET
# ============================================================================


@dataclass
class LibricssDownloadParams(BaseDownloadParams):
    """Download parameters for LIBRICSS dataset"""

    pass


@dataclass
class LibricssProcessParams(BaseProcessParams):
    """Process parameters for LIBRICSS dataset"""

    corpus_dir: Pathlike = None
    type: str = "mdm"
    segmented_cuts: bool = False


# ============================================================================
# LIBRIMIX DATASET
# ============================================================================


@dataclass
class LibrimixDownloadParams(BaseDownloadParams):
    """Download parameters for LIBRIMIX dataset"""

    url: Optional[str] = "https://zenodo.org/record/3871592/files/MiniLibriMix.zip"


@dataclass
class LibrimixProcessParams(BaseProcessParams):
    """Process parameters for LIBRIMIX dataset"""

    librimix_csv: Pathlike = None
    with_precomputed_mixtures: bool = False
    sampling_rate: int = 16000
    min_segment_seconds: float = 3.0


# ============================================================================
# LIBRISPEECH DATASET
# ============================================================================


@dataclass
class LibrispeechDownloadParams(BaseDownloadParams):
    """Download parameters for LIBRISPEECH dataset"""

    dataset_parts: Optional[Union[str, Sequence[str]]] = "mini_librispeech"
    alignments: bool = False
    base_url: str = "http://www.openslr.org/resources"
    alignments_url: str = None  # LIBRISPEECH_ALIGNMENTS_URL


@dataclass
class LibrispeechProcessParams(BaseProcessParams):
    """Process parameters for LIBRISPEECH dataset"""

    corpus_dir: Pathlike = None
    alignments_dir: Optional[Pathlike] = None
    dataset_parts: Union[str, Sequence[str]] = "auto"
    normalize_text: str = "none"


# ============================================================================
# LIBRITTS DATASET
# ============================================================================


@dataclass
class LibrittsDownloadParams(BaseDownloadParams):
    """Download parameters for LIBRITTS dataset"""

    use_librittsr: bool = False
    dataset_parts: Optional[Union[str, Sequence[str]]] = "all"
    base_url: Optional[str] = "http://www.openslr.org/resources"


@dataclass
class LibrittsProcessParams(BaseProcessParams):
    """Process parameters for LIBRITTS dataset"""

    corpus_dir: Pathlike = None
    dataset_parts: Union[str, Sequence[str]] = "all"
    link_previous_utt: bool = False


# ============================================================================
# LIBRITTSR DATASET
# ============================================================================


@dataclass
class LibrittsrDownloadParams(BaseDownloadParams):
    """Download parameters for LIBRITTSR dataset"""

    dataset_parts: Optional[Union[str, Sequence[str]]] = "all"
    base_url: Optional[str] = "http://www.openslr.org/resources"


@dataclass
class LibrittsrProcessParams(BaseProcessParams):
    """Process parameters for LIBRITTSR dataset (alias for LibrittsProcessParams)"""

    corpus_dir: Pathlike = None
    dataset_parts: Union[str, Sequence[str]] = "all"
    link_previous_utt: bool = False


# ============================================================================
# LJSPEECH DATASET
# ============================================================================


@dataclass
class LjspeechDownloadParams(BaseDownloadParams):
    """Download parameters for LJSPEECH dataset"""

    pass


@dataclass
class LjspeechProcessParams(BaseProcessParams):
    """Process parameters for LJSPEECH dataset"""

    corpus_dir: Pathlike = None
    output_dir: Optional[Pathlike] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls - exclude num_jobs for LJSpeech"""
        return {
            k: v for k, v in self.__dict__.items() if v is not None and k != "num_jobs"
        }


# ============================================================================
# MAGICDATA DATASET
# ============================================================================


@dataclass
class MagicdataDownloadParams(BaseDownloadParams):
    """Download parameters for MAGICDATA dataset"""

    base_url: str = "http://www.openslr.org/resources"


@dataclass
class MagicdataProcessParams(BaseProcessParams):
    """Process parameters for MAGICDATA dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# MDCC DATASET
# ============================================================================


@dataclass
class MdccDownloadParams(BaseDownloadParams):
    """Download parameters for MDCC dataset"""

    pass


@dataclass
class MdccProcessParams(BaseProcessParams):
    """Process parameters for MDCC dataset"""

    corpus_dir: Pathlike = None
    dataset_parts: Union[str, Sequence[str]] = "all"


# ============================================================================
# MEDICAL DATASET
# ============================================================================


@dataclass
class MedicalDownloadParams(BaseDownloadParams):
    """Download parameters for MEDICAL dataset"""

    pass


@dataclass
class MedicalProcessParams(BaseProcessParams):
    """Process parameters for MEDICAL dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# MOBVOIHOTWORDS DATASET
# ============================================================================


@dataclass
class MobvoihotwordsDownloadParams(BaseDownloadParams):
    """Download parameters for MOBVOIHOTWORDS dataset"""

    base_url: Optional[str] = "http://www.openslr.org/resources"


@dataclass
class MobvoihotwordsProcessParams(BaseProcessParams):
    """Process parameters for MOBVOIHOTWORDS dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# MTEDX DATASET
# ============================================================================


@dataclass
class MtedxDownloadParams(BaseDownloadParams):
    """Download parameters for MTEDX dataset"""

    languages: Optional[Union[str, Sequence[str]]] = "all"


@dataclass
class MtedxProcessParams(BaseProcessParams):
    """Process parameters for MTEDX dataset"""

    corpus_dir: Pathlike = None
    languages: Optional[Union[str, Sequence[str]]] = "all"


# ============================================================================
# MUSAN DATASET
# ============================================================================


@dataclass
class MusanDownloadParams(BaseDownloadParams):
    """Download parameters for MUSAN dataset"""

    url: Optional[str] = None  # MUSAN_URL


@dataclass
class MusanProcessParams(BaseProcessParams):
    """Process parameters for MUSAN dataset"""

    corpus_dir: Pathlike = None
    parts: Sequence[str] = ("music", "speech", "noise")
    use_vocals: bool = True


# ============================================================================
# NSC DATASET
# ============================================================================


@dataclass
class NscProcessParams(BaseProcessParams):
    """Process parameters for NSC dataset (no download function)"""

    corpus_dir: Pathlike = None
    dataset_part: str = "PART3_SameCloseMic"


# ============================================================================
# PEOPLES SPEECH DATASET
# ============================================================================


@dataclass
class PeoplesSpeechProcessParams(BaseProcessParams):
    """Process parameters for PEOPLES SPEECH dataset (no download function)"""

    corpus_dir: Pathlike = None


# ============================================================================
# PRIMEWORDS DATASET
# ============================================================================


@dataclass
class PrimewordsDownloadParams(BaseDownloadParams):
    """Download parameters for PRIMEWORDS dataset"""

    base_url: str = "http://www.openslr.org/resources"


@dataclass
class PrimewordsProcessParams(BaseProcessParams):
    """Process parameters for PRIMEWORDS dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# SPGISPEECH DATASET
# ============================================================================


@dataclass
class SpgispeechDownloadParams(BaseDownloadParams):
    """Download parameters for SPGISPEECH dataset (manual download only)"""

    pass


@dataclass
class SpgispeechProcessParams(BaseProcessParams):
    """Process parameters for SPGISPEECH dataset"""

    corpus_dir: Pathlike = None
    output_dir: Pathlike  # Required for this dataset
    normalize_text: bool = True


# ============================================================================
# STCMDS DATASET
# ============================================================================


@dataclass
class StcmdsDownloadParams(BaseDownloadParams):
    """Download parameters for STCMDS dataset"""

    base_url: str = "http://www.openslr.org/resources"


@dataclass
class StcmdsProcessParams(BaseProcessParams):
    """Process parameters for STCMDS dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# TEDLIUM DATASET
# ============================================================================


@dataclass
class TedliumDownloadParams(BaseDownloadParams):
    """Download parameters for TEDLIUM dataset"""

    pass


@dataclass
class TedliumProcessParams(BaseProcessParams):
    """Process parameters for TEDLIUM dataset"""

    tedlium_root: Pathlike = None
    dataset_parts: Union[str, Sequence[str]] = None  # TEDLIUM_PARTS
    normalize_text: str = "none"


# ============================================================================
# VCTK DATASET
# ============================================================================


@dataclass
class VctkDownloadParams(BaseDownloadParams):
    """Download parameters for VCTK dataset"""

    use_edinburgh_vctk_url: Optional[bool] = False
    url: Optional[str] = None  # CREST_VCTK_URL


@dataclass
class VctkProcessParams(BaseProcessParams):
    """Process parameters for VCTK dataset"""

    corpus_dir: Pathlike = None
    use_edinburgh_vctk_url: Optional[bool] = False
    mic_id: Optional[str] = "mic2"


# ============================================================================
# VOXCELEB DATASET
# ============================================================================


@dataclass
class Voxceleb1DownloadParams(BaseDownloadParams):
    """Download parameters for VOXCELEB1 dataset"""

    pass


@dataclass
class Voxceleb2DownloadParams(BaseDownloadParams):
    """Download parameters for VOXCELEB2 dataset"""

    pass


@dataclass
class VoxcelebProcessParams(BaseProcessParams):
    """Process parameters for VOXCELEB dataset"""

    voxceleb1_root: Optional[Pathlike] = None
    voxceleb2_root: Optional[Pathlike] = None


# ============================================================================
# REAZONSPEECH DATASET
# ============================================================================


@dataclass
class ReazonspeechDownloadParams(BaseDownloadParams):
    """Download parameters for REAZONSPEECH dataset"""

    dataset_parts: Optional[Union[str, Sequence[str]]] = "auto"
    num_jobs: int = 1


@dataclass
class ReazonspeechProcessParams(BaseProcessParams):
    """Process parameters for REAZONSPEECH dataset"""

    corpus_dir: Pathlike = None
    num_jobs: int = 1


# ============================================================================
# RIR NOISE DATASET
# ============================================================================


@dataclass
class RirNoiseDownloadParams(BaseDownloadParams):
    """Download parameters for RIR NOISE dataset"""

    url: Optional[str] = None  # RIR_NOISE_ZIP_URL


@dataclass
class RirNoiseProcessParams(BaseProcessParams):
    """Process parameters for RIR NOISE dataset"""

    corpus_dir: Pathlike = None
    parts: Sequence[str] = ("point_noise", "iso_noise", "real_rir", "sim_rir")


# ============================================================================
# SPEECHCOMMANDS DATASET
# ============================================================================


@dataclass
class SpeechcommandsDownloadParams(BaseDownloadParams):
    """Download parameters for SPEECHCOMMANDS dataset"""

    speechcommands_version: str = "2"


@dataclass
class SpeechcommandsProcessParams(BaseProcessParams):
    """Process parameters for SPEECHCOMMANDS dataset"""

    speechcommands_version: str = "2"
    corpus_dir: Pathlike = None
    output_dir: Optional[Pathlike] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls - exclude num_jobs for SpeechCommands"""
        return {
            k: v for k, v in self.__dict__.items() if v is not None and k != "num_jobs"
        }


# ============================================================================
# THCHS_30 DATASET
# ============================================================================


@dataclass
class Thchs30DownloadParams(BaseDownloadParams):
    """Download parameters for THCHS_30 dataset"""

    base_url: str = "http://www.openslr.org/resources"


@dataclass
class Thchs30ProcessParams(BaseProcessParams):
    """Process parameters for THCHS_30 dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# THIS AMERICAN LIFE DATASET
# ============================================================================


@dataclass
class ThisAmericanLifeDownloadParams(BaseDownloadParams):
    """Download parameters for THIS AMERICAN LIFE dataset"""

    metadata_url: None = "https://ipfs.io/ipfs/bafybeidyt3ch6t4dtu2ehdriod3jvuh34qu4pwjyoba2jrjpmqwckkr6q4/this_american_life.zip"
    website_url: None = "https://thisamericanlife.org"


@dataclass
class ThisAmericanLifeProcessParams(BaseProcessParams):
    """Process parameters for THIS AMERICAN LIFE dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# TIMIT DATASET
# ============================================================================


@dataclass
class TimitDownloadParams(BaseDownloadParams):
    """Download parameters for TIMIT dataset"""

    base_url: str = "http://www.openslr.org/resources"


@dataclass
class TimitProcessParams(BaseProcessParams):
    """Process parameters for TIMIT dataset"""

    corpus_dir: Pathlike = None
    num_phones: int = 48


# ============================================================================
# UWB ATCC DATASET
# ============================================================================


@dataclass
class UwbAtccDownloadParams(BaseDownloadParams):
    """Download parameters for UWB ATCC dataset"""

    pass


@dataclass
class UwbAtccProcessParams(BaseProcessParams):
    """Process parameters for UWB ATCC dataset"""

    corpus_dir: Pathlike = None
    silence_sym: Optional[str] = ""
    breath_sym: Optional[str] = ""
    noise_sym: Optional[str] = ""
    foreign_sym: Optional[str] = "<unk>"
    partial_sym: Optional[str] = "<unk>"
    unintelligble_sym: Optional[str] = "<unk>"
    unknown_sym: Optional[str] = "<unk>"


# ============================================================================
# VOXCONVERSE DATASET
# ============================================================================


@dataclass
class VoxconverseDownloadParams(BaseDownloadParams):
    """Download parameters for VOXCONVERSE dataset"""

    corpus_dir: Pathlike = None


@dataclass
class VoxconverseProcessParams(BaseProcessParams):
    """Process parameters for VOXCONVERSE dataset"""

    corpus_dir: Pathlike = None
    split_test: bool = False
    sampling_rate: Optional[int] = None


# ============================================================================
# VOXPOPULI DATASET
# ============================================================================


@dataclass
class VoxpopuliDownloadParams(BaseDownloadParams):
    """Download parameters for VOXPOPULI dataset"""

    subset: Optional[str] = "asr"


@dataclass
class VoxpopuliProcessParams(BaseProcessParams):
    """Process parameters for VOXPOPULI dataset"""

    corpus_dir: Pathlike = None
    task: str = "asr"
    lang: str = "en"
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None


# ============================================================================
# XBMU AMDO31 DATASET
# ============================================================================


@dataclass
class XbmuAmdo31DownloadParams(BaseDownloadParams):
    """Download parameters for XBMU AMDO31 dataset"""

    pass


@dataclass
class XbmuAmdo31ProcessParams(BaseProcessParams):
    """Process parameters for XBMU AMDO31 dataset"""

    corpus_dir: Pathlike = None


# ============================================================================
# EGO4D DATASET
# ============================================================================


@dataclass
class Ego4dDownloadParams(BaseDownloadParams):
    """Download parameters for EGO4D dataset"""

    # AWS credentials are loaded from environment variables:
    # AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
    dataset_parts: Optional[List[str]] = None  # Defaults to clips and annotations
    install_cli: bool = True  # Whether to install Ego4D CLI if not available
    timeout: int = 3600  # Timeout for download operations in seconds
    env_file: Optional[Pathlike] = None  # Path to .env file for environment variables


@dataclass
class Ego4dProcessParams(BaseProcessParams):
    """Process parameters for EGO4D dataset"""

    corpus_dir: Pathlike = None
    extract_audio: bool = True  # Extract audio from video files
    audio_sample_rate: int = 16000  # Sample rate for extracted audio
    min_segment_duration: float = 0.5  # Minimum segment duration in seconds
    max_segment_duration: float = 30.0  # Maximum segment duration in seconds
    max_clips: int = 0  # 0 means no limit; otherwise limit number of clips
    annotation_subset: Optional[str] = None  # e.g., "av" to target AV annotations


# ============================================================================
# YESNO DATASET
# ============================================================================


@dataclass
class YesnoDownloadParams(BaseDownloadParams):
    """Download parameters for YESNO dataset"""

    url: Optional[str] = None


@dataclass
class YesnoProcessParams(BaseProcessParams):
    """Process parameters for YESNO dataset"""

    corpus_dir: Pathlike = None
    output_dir: Optional[Pathlike] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for function calls - exclude num_jobs for YESNO"""
        return {
            k: v for k, v in self.__dict__.items() if v is not None and k != "num_jobs"
        }


# ============================================================================
# MSDWILD DATASET
# ============================================================================


@dataclass
class MswildDownloadParams(BaseDownloadParams):
    """Download parameters for MSDWILD dataset"""

    download_audio: bool = True
    download_video: bool = False
    download_faces: bool = False


@dataclass
class MswildProcessParams(BaseProcessParams):
    """Process parameters for MSDWILD dataset"""

    corpus_dir: Pathlike = None
    splits: Optional[Dict[str, str]] = None
    sampling_rate: int = Optional[int]


# ============================================================================
# AVA-AVD DATASET
# ============================================================================


@dataclass
class AvaAvdDownloadParams(BaseDownloadParams):
    """Download parameters for AVA-AVD dataset"""

    download_annotations: bool = True
    download_videos: bool = True


@dataclass
class AvaAvdProcessParams(BaseProcessParams):
    """Process parameters for AVA-AVD dataset"""

    corpus_dir: Pathlike = None
    splits: Optional[Dict[str, str]] = None
    sampling_rate: int = 16000  # Target sampling rate for audio extraction


# ============================================================================
# LIBRIHEAVY MIX DATASET
# ============================================================================


@dataclass
class LibriheavyMixDownloadParams(BaseDownloadParams):
    """Download parameters for LibriheavyMix dataset"""

    dataset_parts: Union[str, List[str]] = "small"
    speaker_counts: Union[int, List[int]] = field(
        default_factory=lambda: [1, 2, 3, 4]
    )  # Number of speakers per mixture
    cache_dir: Optional[Pathlike] = None


@dataclass
class LibriheavyMixProcessParams(BaseProcessParams):
    """Process parameters for LibriheavyMix dataset"""

    corpus_dir: Pathlike = None
    dataset_parts: Union[str, List[str]] = "small"
    speaker_counts: Union[int, List[int]] = field(
        default_factory=lambda: [1, 2, 3, 4]
    )  # Number of speakers per mixture
    splits: Optional[Dict[str, str]] = None
    min_speakers: int = 1  # Minimum number of speakers
    max_speakers: int = 4  # Maximum number of speakers
    sampling_rate: Optional[int] = None  # Target sampling rate for audio

