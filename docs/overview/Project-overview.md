## TS-DIA: Ego‑centric Speaker Diarization (Project Overview)

A research prototype for person‑centric, auto‑regressive speaker diarization that avoids the speaker permutation problem and scales to unknown numbers of speakers.

### Why TS-DIA?
- **Person‑centric labeling**: predict one target at a time with five exclusive classes: `ts`, `ts_ovl`, `others_sgl`, `others_ovl`, `ns`.
- **No permutation problem**: iterative decoding converts ego‑centric outputs into global “who‑spoke‑when”.
- **Extensible**: built around a Transformer encoder–decoder design (implementation in progress).
- **Dataset‑agnostic**: unified preprocessing across diverse diarization datasets.

### Repo status
- Documentation and dataset notes are available under `docs/`.
- Code skeleton exists (e.g., `train.py`, `infrance.py`); core model and metrics are being implemented.

### Key ideas (at a glance)
- Reframe diarization as: “Is the target speaker talking now (and is there overlap)?”
- Use mutually exclusive, time‑dependent classes to better capture speaking style and turn‑taking.
- Decode iteratively over detected speakers to assemble global diarization.

### Datasets
Start here:
- Overview: `docs/overview/Dataset Selection.md`
- Cards: `docs/datasets/` (e.g., `AMI Dataset (information card).md`, `ICSI dataset overview.md`, `AISHELL-4 dataset overview.md`, `VoxConverse dataset overview.md`, `Earnings-21.md`, `AVA-AVD.md`, `MSDWild.md`, `LibriheavyMix.md`, `Ego4D Audio Diarization.md`)

Notes
- Simulated mixtures (e.g., LibriheavyMix) are great for pretraining scale but lack natural turn‑taking; fine‑tune on real dialog.
- Some datasets may be excluded depending on access, documentation, and label quality.

### Roadmap
- [ ] Data preprocessing to unified ego‑centric labels
- [ ] Transformer encoder–decoder implementation
- [ ] Training loop and configuration
- [ ] Iterative decoding → global diarization
- [ ] Evaluation (DER/JER) scripts
- [ ] Example notebooks and demos

### Contributing
Contributions are welcome! Please:
- Open an issue to discuss significant changes
- Use `pre-commit` hooks (`pip install -r requirements-dev.txt` and `pre-commit install`)

### License
This project is released under the terms of the license in `LICENSE`.

### Acknowledgements
Developed in collaboration with Språkbanken and KTH. Mentorship by Jens Edlund (Språkbanken Tal; Division of Speech, Music and Hearing, KTH).