from torch import nn
from pathlib import Path
from typing import Union, Literal, Any
from model.transformer import EncoderDecoderTransformer
import torch


def collect_model(checkpoint: Union[Path, str], model: Union[EncoderDecoderTransformer, nn.Module], device: Union[Any, Literal['cpu', 'cuda']]) -> Union[EncoderDecoderTransformer, nn.Module]:
    print(f"Loading checkpoint from {checkpoint}")
    if str(checkpoint).endswith(".safetensors"):
        from safetensors.torch import load_file
        from model.utils import fix_safetensors_shared_parameters

        state_dict = load_file(checkpoint)
        state_dict = fix_safetensors_shared_parameters(state_dict, model)

        model.load_state_dict(state_dict)
    else:
        checkpoint_obj = torch.load(checkpoint, map_location=device)
        if "model_state_dict" in checkpoint_obj:
            model.load_state_dict(checkpoint_obj["model_state_dict"])
        else:
            model.load_state_dict(checkpoint_obj)

    return model
