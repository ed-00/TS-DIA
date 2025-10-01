To-Do List: Diarization Dataset Preparation
For datasets already supported with Lhotse recipes:

[] AMI Corpus

[] ICSI Meeting Corpus

[] AISHELL-4

[] Earnings-21

Actions:

Use Lhotse built-in recipes to download, preprocess, and generate manifests.

For datasets requiring custom manifest scripts:

[] MSDWild

[] VoxConverse

[] AVA-AVD

[] LibriheavyMix

[] M3SD

[] Ego4D Audio Diarization

Actions:

Write custom scripts to:

Download dataset audio and label files.

Parse diarization/segment annotations (RTTM, CSV, JSON, etc.).

Convert to Lhotse’s manifest format (or, for AV data, consider extension).

(Optional) Contribute new Lhotse recipe if workflow is generally useful