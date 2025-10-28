"""Dataset class for Ego-centric diarization tasks."""
from dataclasses import dataclass
from torch import Tensor
from torch.utils.data import Dataset
from lhotse import CutSet, SupervisionSet, validate, RecordingSet


@dataclass
class EgoCentricLables:
    """Ego-centric lables 
    """
    features: Tensor  # (B, F, T)
    # (B, C) C /in {non_speech, ts, ts_ovl, other_sgl, other_ovl}
    speaker_activity: Tensor
    ts: Tensor  # (B, F)


class EgoDataset(Dataset):
    """Ego-centric class for diarization 

        returns a feature dict: 
    """

    def __init__(self, cuts: CutSet, uem: SupervisionSet):
        """_summary_

        Args:
            cutsets (CutSet): _description_
            uem (Optional[SupervisionSet], optional): _description_. Defaults to None.
            global_speaker_ids (bool, optional): _description_. Defaults to False.
        """
        super().__init__()
        validate(cuts)

        recordings = RecordingSet(
            {
                c.recording.id: c.recording for c in cuts if c.has_recording
            }
        )
        
        uem_intervals = CutSet.from_manifests(
            recordings=recordings,
            supervisions=uem
        ).index_supervisions()
        
        
        
        
        # Get the speakers
        
        # for speakers in 
        

    def __getitem__(self, index) -> EgoCentricLables:
        return {

        }
