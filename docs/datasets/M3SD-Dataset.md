Here are direct quotes from the paper “M3SD: Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset” matching your questions:

---

**1. Dataset Overview**  
a. **Total size (hours, segments):**  
“The dataset contains 1,372 records, 770+ hours of data, and a large number of different speakers.”  
“This dataset contains 770+ hours and 1372 segments of conversations...”

b. **Domains and scenarios:**  
“This dataset ... including many scenarios such as online/offline meetings, debates, speeches, home/outdoor conversations, movies, and news broadcasts.”

---

**2. Recording Conditions**  
a. **Audio quality:**  
“In this article, we use the FFmpeg toolkit to extract it, and get the audio with a sampling rate of 44k and the video with a resolution of 720p.”  
“Deep Noise Suppression Mean Opinion Score (DNSMOS) ... generating three key scores: speech quality (SIG), background noise quality (BAK), and overall audio quality (OVRL), with a score of 0-5 (5 being the best). ... this paper sets the score threshold to 3 points.”

b. **Microphone configurations:**  
The data is sourced from web videos, so typical setups include:  
“We crawl video data on the video websites YouTube and BiliBili... crawl video data in different scenes and different languages...”  
No explicit single/multi-microphone hardware config; data variety implied by online/video sources.

---

**3. Speaker & Language Characteristics**  
a. **Speakers per recording:**  
Not explicitly quantified, but diverse:  
“...multiple participants... highly diverse. ... a large number of different speakers.”  
“...multi-person debates, outdoor conversations, etc.”

b. **Languages represented:**  
“...including Chinese, English, Japanese, and other languages, etc.”  
“...expand the diversity of the dataset, so as to improve the generalization ability of subsequent training models.”

---

**4. Annotations & Metadata**  
a. **Annotation types:**  
“...generating pseudo-labels using pre-trained audio-only as well as audiovisual speaker diarization models, without the need for manual annotation.”  
Implied: speaker turns, per-frame speech activity, overlap via EEND.

b. **Annotation format:**  
Not stated directly; likely model output format.  
“...speaker diarization pseudo-labels.”  
Check code for specifics (see Licensing & Access).

c. **Forced alignment or transcript segmentation:**  
No explicit mention of forced alignment/transcript segmentation.  
“...fine-grained annotation work is often time-consuming and costly.... pseudo-labels ... can make pseudo-labels more accurate through iterative training.”

---

**5. Licensing & Access**  
a. **Licensing model:**  
“Our dataset and code have been open-sourced at [https://huggingface.co/spaces/OldDragon/m3sd.”](https://huggingface.co/spaces/OldDragon/m3sd.%E2%80%9D)

b. **Download links:**  
“Our dataset and code have been open-sourced at [https://huggingface.co/spaces/OldDragon/m3sd.”](https://huggingface.co/spaces/OldDragon/m3sd.%E2%80%9D)

---

**6. Data Splits & Benchmarks**  
a. **Splits:**  
No explicit split structure described; likely available in code/scripts.

b. **Evaluation metrics:**  
DER (Diarization Error Rate) is described:  
“In the clustering stage, ... aggregate the regions of each speaker into separate clusters. Some systems have achieved good performance...”  
The DOVER-LAP fusion method is mentioned:  
“...DOVER-LAP to fuse the results ... more accurate pseudo-labels.”

c. **Baseline results:**  
No explicit baseline numbers quoted for this dataset in the text; methods/framework cited.

---

**7. Usage & Preprocessing**  
a. **Preprocessing steps:**  
“It is necessary to carry out a quality evaluation of audio and video.... speech quality assessment, video quality assessment, and audio-visual synchronization detection.”  
“...preprocess the audio and video data, which mainly includes face detection, face trajectory tracking, and extraction of the lip region of interest (ROI). Then, ... voting fusion to integrate the output results of the two models to obtain the speaker diarization pseudo-labels.”

b. **Toolkits/scripts:**  
Mentioned:  
“The data cleaning process in this article ... PySceneDetect toolkit ... FFmpeg toolkit ... DNSMOS ... MD-VQA ... SyncNet ... RetinaFace ... DeepSORT ... MediaPipe ... DOVER-LAP ... 3D-Speaker multi-speaker diarization network.”

---

**8. Challenges & Limitations**  
a. **Known challenges:**  
“...building a high-quality speaker diarization dataset requires a lot of manpower and material resources... Especially when dealing with complex scenarios such as multiperson interaction, speech overlap, and strong background noise, ... annotation difficulty and cost are too high.”  
“...complex acoustic scenes, a large number of speech overlaps and domain mismatches...”

b. **Limitations:**  
“...the existing datasets are not enough for the future development... need to build larger-scale data resources that include more scenarios and more languages.”  
License: “open-sourced” but data from YouTube/BiliBili—check link for details.

---

**9. Research Applications**  
a. **Research tasks:**  
“...promote the implementation of the whole speech processing system...”  
“...speaker diarization ... speaker embedding extraction ... clustering ... audio-visual speaker diarization (AVSD) ... overlap detection.”

b. **Notable papers:**  
Several references cited for systems and datasets; see full reference list for specifics.

---

**10. Reproducibility**  
a. **Reproducibility:**  
“The dataset and code have been open-sourced ... and we have open-sourced the code for data collection to facilitate the construction of larger datasets.”

b. **Version controls / update logs:**  
No explicit mention of version control/update logs in the paper; check HuggingFace repo for repository details.

---

If you need more verbatim quotes for a specific item or code/annotation samples, let me know!

1. [https://arxiv.org/pdf/2506.14427v2.pdf](https://arxiv.org/pdf/2506.14427v2.pdf)