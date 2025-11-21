#!/usr/bin/env python3
"""Reproduce NaN from diagnostic .pt file.

This script attempts to load a diagnostic file created by Trainer (e.g.
diag_nan_batches/bad_batch_epoch0_step1743_batch1743.pt), build the model
from a config YAML (default uses REPRODUCE/02_pretraining_softmax.yaml),
optionally load weights, run a forward (and optional backward) and report
where NaNs appear.

Example:
  python scripts/repro_from_diag.py \
    outputs/checkpoints/softmax_pretraining_large_pos/diag_nan_batches/bad_batch_epoch0_step1743_batch1743.pt \
    --config configs/REPRODUCE/02_pretraining_softmax.yaml \
    --device cpu --do-backward

This is intentionally lightweight and defensive so you can run it inside the
running Docker container to quickly inspect saved faulty batches.
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from model.parse_model_args import parse_model_config
from model.model_factory import create_model


def safe_print_tensor(name, t: torch.Tensor):
    arr = t.detach().cpu().numpy()
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, finite={np.isfinite(arr).all()}")
    if not np.isfinite(arr).all():
        print(f"  counts: finite={int(np.isfinite(arr).sum())}, nan={int(np.isnan(arr).sum())}, inf={int(np.isinf(arr).sum())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diag", type=Path, help="Path to diagnostic .pt file saved by Trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/REPRODUCE/02_pretraining_softmax.yaml"), help="Path to model YAML (used to instantiate same architecture)")
    parser.add_argument("--weights", type=Path, default=None, help="Optional model state_dict or .pt checkpoint to load (if present)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--amp", action="store_true", help="Run forward in AMP (torch.cuda.amp.autocast) to mimic mixed precision")
    parser.add_argument("--do-backward", action="store_true", help="Compute a backward pass and inspect gradients")
    parser.add_argument("--save-out", type=Path, default=None, help="Where to save local debug outputs (.pt)")
    args = parser.parse_args()

    if not args.diag.exists():
        raise SystemExit(f"Diag file not found: {args.diag}")

    print(f"Loading diag file: {args.diag}")
    data = torch.load(str(args.diag), map_location="cpu")

    # Basic checks
    if not isinstance(data, dict):
        raise SystemExit("Diag file did not contain expected dict structure")

    # Extract tensors
    features = data.get("features")
    labels = data.get("labels")

    if features is None:
        raise SystemExit("No 'features' key in diag file")

    if labels is None:
        print("No 'labels' found — will forward-only and inspect outputs")

    print("Basic diag keys:")
    for k in data.keys():
        v = data[k]
        print(f" - {k}: {type(v)}")

    safe_print_tensor("features", features)
    if labels is not None and isinstance(labels, torch.Tensor):
        safe_print_tensor("labels", labels)

    # Build model from config
    print(f"Loading model config: {args.config}")
    model_config = parse_model_config(args.config)
    model = create_model(model_config)

    device = torch.device(args.device)
    model.to(device)

    # Optionally load weights if provided
    if args.weights is not None:
        if not args.weights.exists():
            print(f"Warning: weights file not found: {args.weights}")
        else:
            print(f"Attempting to load weights from: {args.weights}")
            # Support .safetensors files and plain torch state_dicts
            try:
                if args.weights.suffix == ".safetensors":
                    # load with safetensors if available
                    from safetensors.torch import load_file as st_load

                    sdata = st_load(str(args.weights), device=device)
                    # safetensors returns dict[str, torch.Tensor]; try to load directly
                    try:
                        model.load_state_dict(sdata, strict=False)
                        print("Loaded weights from safetensors (non-strict)")
                    except Exception as e:
                        print(f"safetensors load -> load_state_dict error: {e}. Will try mapping keys.")
                        # try converting to cpu tensors and loading
                        mapped = {k: v.to("cpu") if hasattr(v, 'to') else torch.tensor(v) for k, v in sdata.items()}
                        try:
                            model.load_state_dict(mapped, strict=False)
                            print("Loaded safetensors weights after CPU mapping")
                        except Exception as e2:
                            print(f"Failed to map safetensors state to model: {e2}")
                else:
                    ckpt = torch.load(str(args.weights), map_location=device)
                    # Try common keys
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        model.load_state_dict(ckpt["model_state_dict"], strict=False)
                        print("Loaded model_state_dict from checkpoint dict")
                    else:
                        try:
                            model.load_state_dict(ckpt)
                            print("Loaded state_dict directly")
                        except Exception as e:
                            # If ckpt seems to have nested accelerate files (e.g., 'state' folder), try heuristics
                            print(f"Failed to load weights with torch.load: {e}")
            except Exception as ex:
                print(f"Exception while loading weights: {ex}")

    model.train()  # replicate training-time forward/backward behavior

    x = features.to(device)
    y = labels.to(device) if labels is not None else None

    try:
        if args.amp and device.type == "cuda":
            autocast = torch.cuda.amp.autocast
        else:
            # no-op context manager
            class _noop:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            autocast = _noop

        with autocast():
            print("Running forward() ...")
            out = model(x=x, is_target=None, labels=y if y is not None else None)

        # Get logits tensor
        logits = out if isinstance(out, torch.Tensor) else getattr(out, "logits", None)

        if logits is None:
            print("Warning: model did not return logits tensor; object repr:")
            print(repr(out)[:1000])
        else:
            safe_print_tensor("logits", logits)

        # If logits contain NaNs report and optionally stop
        if logits is not None and not torch.isfinite(logits).all():
            print("Detected NaNs/Inf in logits — forward failed. Details above.")
        else:
            print("Forward produced finite logits")

        if args.do_backward:
            if logits is None or y is None:
                print("Skipping backward because logits or labels are missing")
            else:
                # follow training: ignore first token in sequence when computing loss
                try:
                    l = logits[:, 1:, :]
                    t = y
                    if t.dim() == l.dim() - 1:
                        # t is [batch, seq] -> use standard cross entropy
                        num_classes = l.size(-1)
                        loss = F.cross_entropy(l.reshape(-1, num_classes), t.reshape(-1), reduction="mean")
                    else:
                        # fallback to MSE on-floating targets
                        loss = F.mse_loss(l, t, reduction="mean")
                    print("Loss scalar:", float(loss.detach().cpu()))

                    # backward
                    loss.backward()

                    # inspect gradients for NaN/Inf
                    invalid_grad = False
                    for i, p in enumerate(model.parameters()):
                        if p.grad is not None:
                            g = p.grad.detach().cpu()
                            if not np.isfinite(g.numpy()).all():
                                invalid_grad = True
                                print(f"NaN/Inf gradient in parameter idx {i}, shape={g.shape}")
                                break
                    if not invalid_grad:
                        print("Backward produced finite gradients")
                except Exception as e:
                    print("Backward exception:", e)

    except Exception as e:
        print("Forward exception:", e)

    if args.save_out is not None:
        out_path = args.save_out
        print(f"Saving debug snapshot to: {out_path}")
        torch.save({"diag": data, "logits": logits.detach().cpu() if logits is not None else None}, str(out_path))


if __name__ == "__main__":
    main()
