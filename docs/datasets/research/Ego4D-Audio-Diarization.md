**Ego4D Audio Diarization**  
    Lin, T. H., et al. “Ego4D: Audio-only Speaker Diarization in First-Person Video.” *CVPR Workshops 2023*.  
    https://ego4d-data.org/  

---

## 1. **Dataset Overview**

**a. Total size (hours, segments):**  
"EGO4D is the world's largest egocentric (first person) video ML dataset and benchmark suite, with 3,600 hrs (and counting) of densely narrated video..."

- "Full Primary Dataset: ~7.1 TB"
    
- "Entire Dataset: 30+ TB"
    
- "Benchmark Clips: ~1 TB"  
    arxiv+1
    

**b. Domains and scenarios:**  
"It covers hundreds of scenarios (household, outdoor, workplace, leisure, etc.) of daily life activity captured in-the-wild by 926 unique camera wearers from 74 worldwide locations and 9 different countries."  
"Portions of the video are accompanied by audio, 3D meshes of the environment, eye gaze, stereo, and/or synchronized videos from multiple egocentric cameras at the same event."  
[arxiv](https://arxiv.org/abs/2110.07058)

---

## 2. **Recording Conditions**

**a. Audio quality (sampling rate, noise level):**  
"Audio stream ... Audio rate (32KHz vs 48KHz)."  
"Audio set to AAC" for canonical videos.  
[ego4d-data](https://ego4d-data.org/docs/data/videos/)

**b. Microphone configurations:**  
"Audio channel layout (mono vs. audio)"  
[ego4d-data](https://ego4d-data.org/docs/data/videos/)

---

## 3. **Speaker & Language Characteristics**

**a. Speakers per recording (average, range):**  
Annotations include "Person IDs for each Face Track in video clip" aka anonymous speaker labels; large group settings as well as 1:1 and casual conversational scenes are represented (see Social and AVS benchmarks).  
[ego4d-data](https://ego4d-data.org/docs/data/annotation-guidelines/)

**b. Languages represented and distribution:**  
"Dense written sentence narrations in English..."  
Other spoken languages are present due to international sites but narration and most transcriptions are in English.  
ego4d-data+1

---

## 4. **Annotations & Metadata**

**a. Annotation types:**  
"Speaker Labeling and AV anchor extraction: Anonymous Person IDs for each Face Track in video clip"  
"Speech Segmentation (Per Speaker): Temporal segments for voice activity for the camera wearer and for each Person ID"  
"Transcription: Video clip audio transcriptions"  
"Corrected Speech Transcription annotations matching voice activity segments and Person IDs from AV2"  
[ego4d-data](https://ego4d-data.org/docs/data/annotation-guidelines/)

**b. Annotation format:**  
"Once you download the annotations with the CLI, you'll have a set of json files. Here are their schemas for quick reference..."  
"Additional annotation formats ... for a unified json across the FHO tasks, which is now available."  
Mention of RTTM, JSON, and TSV in the schemas and benchmark documentation.  
ego4d-data+1

**c. Forced alignment / transcript segmentation:**  
"Corrected Speech Transcription annotations matching voice activity segments and Person IDs from AV2"  
No explicit mention of forced alignment; segmentation is available per speaker per segment.  
[ego4d-data](https://ego4d-data.org/docs/data/annotation-guidelines/)

---

## 5. **Licensing & Access**

**a. Licensing model:**  
"Obtaining the dataset or any annotations requires you first review our license agreement and accept the terms. Go here (ego4ddataset.com) to review and execute this agreement, and you will be emailed a set of AWS access credentials when your license agreement is approved, which will take ~48hrs."  
ego4ddataset+1

**b. Download links:**  
"Run the CLI to download the dataset"  
Example:  
`ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations`  
github+1

---

## 6. **Data Splits & Benchmarks**

**a. Training/dev/test splits:**  
"FHO's train/val/test split is partitioned by 5 minute clips..."  
"NLQ: average length is 10 minutes, clips are at most 20 minutes long..."  
"AV: roughly 5 minute intervals."  
[ego4d-data](https://ego4d-data.org/docs/data/videos/)

**b. Evaluation metrics:**  
"Evaluation metrics and provide benchmark results for three tasks on the dataset: part mask segmentation, object and part attribute prediction and zero-shot instance detection."  
DER, JER, etc. mentioned in the audio/visual diarization docs.  
arxiv+1

**c. Baseline results published:**  
"Baseline model we call EgoSTARK. We publicly release our annotations and benchmark, hoping our dataset leads to further advancements in tracking."  
See challenge and updates documentation.  
[ego4d-data](https://ego4d-data.org/docs/data/egotracks/)

---

## 7. **Usage & Preprocessing**

**a. Preprocessing steps:**  
"Canonical videos are normalized ... 30FPS, Sample Aspect Ratio (SAR) 1:1, Audio set to AAC..."  
"Each annotation file has their fields prefixed ... if it is referring to a canonical clip time/frame or ... canonical video time/frame."  
FFmpeg recommended for remuxing and compression.  
[ego4d-data](https://ego4d-data.org/docs/data/videos/)

**b. Toolkits/scripts for data loading:**  
"Ego4D CLI can be installed via pip and provides access to the Ego4D datasets."  
"Dataloaders ... available for each of the benchmarks"  
ego4d-data+1

---

## 8. **Challenges & Limitations**

**a. Known challenges:**  
"Portions of the video are accompanied ... robust de-identification procedures where relevant."  
Section on "Erroneous Videos Removed From Dataset & Benchmarks: ... 1 video with frozen frames, 1 with varying resolution, and several videos too short ... stereo videos ... removed ... flagged in metadata."  
Reports of domain mismatch and challenge forum posts on annotation errors.  
discuss.ego4d-data+1

**b. Limitations before training:**  
"Once approved your access credentials will expire in 14 days - you're expected to download the data locally, not to consume it from AWS."  
[ego4d-data](https://ego4d-data.org/docs/start-here/)

---

## 9. **Research Applications**

**a. Suited research tasks:**  
"Benchmark tasks: Episodic Memory (queryable video), Hands & Objects, Forecasting, Audio-Visual Diarization, Social Interactions"  
"Visual object tracking ... egocentric vision problems ..."  
ego4d-data+1

**b. Notable papers / findings:**  
"Ego4D: Around the World in 3,000 Hours of Egocentric Video” [arXiv:2110.07058]"  
"PACO: Parts and Attributes of Common Objects" [arXiv:2301.01795]  
"EgoTracks: ... Visual Object Tracking" [arXiv:2301.03213]  
arxiv+2

---

## 10. **Reproducibility**

**a. Can the dataset be reproduced?**  
"Each dataset contains a manifest.csv file ... the master metadata file at ~/ego4d_data/v1/ego4d.json."  
Versioned, with CLI and manifests provided.  
[github](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md)

**b. Version controls / update logs:**  
"Details below." (Updates page, changelog)  
"Specific uids removed are provided in the changelog and the ego4d.json metadata has been updated appropriately."  
github+1

---

These quotes directly reflect the content available in the Ego4D documentation and reference material. For exhaustive details and precise statistics per benchmark, consult the dataset paper and CLI metadata files after access.

1. [https://arxiv.org/abs/2110.07058](https://arxiv.org/abs/2110.07058)
2. [https://ego4d-data.org/docs/start-here/](https://ego4d-data.org/docs/start-here/)
3. [https://ego4d-data.org/docs/data/videos/](https://ego4d-data.org/docs/data/videos/)
4. [https://ego4d-data.org/docs/data/annotation-guidelines/](https://ego4d-data.org/docs/data/annotation-guidelines/)
5. [https://ego4d-data.org/docs/data/annotations-schemas/](https://ego4d-data.org/docs/data/annotations-schemas/)
6. [https://ego4d-data.org/docs/tutorials/FHO_Overview/](https://ego4d-data.org/docs/tutorials/FHO_Overview/)
7. [https://ego4ddataset.com/](https://ego4ddataset.com/)
8. [https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md)
9. [https://ego4d-data.org/docs/CLI/](https://ego4d-data.org/docs/CLI/)
10. [https://arxiv.org/abs/2301.01795](https://arxiv.org/abs/2301.01795)
11. [https://ego4d-data.org/docs/data/egotracks/](https://ego4d-data.org/docs/data/egotracks/)
12. [https://discuss.ego4d-data.org/](https://discuss.ego4d-data.org/)
13. [https://github.com/facebookresearch/Ego4d/blob/main/CHANGELOG](https://github.com/facebookresearch/Ego4d/blob/main/CHANGELOG)
14. [https://ego4d-data.org/docs/benchmarks/overview/](https://ego4d-data.org/docs/benchmarks/overview/)
15. [https://arxiv.org/abs/2301.03213](https://arxiv.org/abs/2301.03213)
16. [https://ego4d-data.org/docs/data/unprocessed_data/](https://ego4d-data.org/docs/data/unprocessed_data/)
17. [https://ego4d-data.org/docs/](https://ego4d-data.org/docs/)
18. [https://ego4d-data.org/docs/viz/](https://ego4d-data.org/docs/viz/)
19. [https://ego4d-data.org/docs/contact/](https://ego4d-data.org/docs/contact/)
20. [https://ffmpeg.org/ffmpeg-formats.html#Options](https://ffmpeg.org/ffmpeg-formats.html#Options)
21. [https://ego4d-data.org/docs/FAQ/#my-credentials-expired--how-do-i-renew](https://ego4d-data.org/docs/FAQ/#my-credentials-expired--how-do-i-renew)
22. [https://github.com/facebookresearch/Ego4d/blob/main/viz/narrations/README.md](https://github.com/facebookresearch/Ego4d/blob/main/viz/narrations/README.md)
23. [https://ego4d-data.org/pdfs/Ego4D-Privacy-and-ethics-consortium-statement.pdf](https://ego4d-data.org/pdfs/Ego4D-Privacy-and-ethics-consortium-statement.pdf)
24. [mailto:privacy@ego4d-data.org](mailto:privacy@ego4d-data.org)
25. [https://github.com/EGO4D/episodic-memory/issues/14](https://github.com/EGO4D/episodic-memory/issues/14)
26. [https://github.com/facebookresearch/paco](https://github.com/facebookresearch/paco)
27. [https://ego4d-data.org/docs/data/features/](https://ego4d-data.org/docs/data/features/)
28. [https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html)
29. [https://github.com/facebookresearch/Ego4d/issues?q=is%3Aissue](https://github.com/facebookresearch/Ego4d/issues?q=is%3Aissue)