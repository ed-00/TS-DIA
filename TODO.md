To-Do List: Diarization Dataset Preparation
For datasets already supported with Lhotse recipes:

[x] AMI Corpus

[x] ICSI Meeting Corpus

[x] AISHELL-4

[x] Earnings-21

Actions:

Use Lhotse built-in recipes to download, preprocess, and generate manifests.

For datasets requiring custom manifest scripts:

[x] MSDWild

[x] VoxConverse

[x] AVA-AVD

[x] LibriheavyMix

[] M3SD

[] Ego4D Audio Diarization

Actions:

Write custom scripts to:

Download dataset audio and label files.

Parse diarization/segment annotations (RTTM, CSV, JSON, etc.).

Convert to Lhotse’s manifest format (or, for AV data, consider extension).

(Optional) Contribute new Lhotse recipe if workflow is generally useful