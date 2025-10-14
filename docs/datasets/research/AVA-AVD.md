**AVA-AVD**  
   Xu, Z., Song, Y., Huang, J., et al. “AVA-AVD: Audio-Visual Speaker Diarization in the Wild.” *ACM Multimedia 2021*.  
   https://arxiv.org/abs/2111.14448  

---

## 1. **Dataset Overview**

**a. Total size (hours, segments):**

- “...AVA-AVD has various complicated scenarios with a wide spectrum of daily activities... around 1500 unique identities with voice tracks and/or faces. ...243 clips (20.25 hours) for training, 54 clips (4.5 hours) for validation, and 54 clips (4.5 hours) for testing...total duration: 29h15m”  
    **Table 1:** “…351 [clips], 29h15m, 1500 [identities], 2/7.7/24 [min/avg/max speakers per clip]…”
    

**b. Domains and scenarios:**

- “...includes in-the-wild scenarios and completely off-screen speakers, e.g., scene from a documentary film.”
    
- “…wide spectrum of daily activities…it provides both a challenging test set and diverse training data.”
    
- Table 1: “...diverse daily activities…”
    

---

## 2. **Recording Conditions**

**a. Audio quality (sampling rate, noise level):**

- “...the input speech segments are 2s and sub-sampled to 16 kHz. The spectrogram is extracted every 10ms with a window of 25ms on the fly...”
    
- “...challenging due to the noise (voices from crowds, traffic noise, etc.), which is not observed indoors…”
    

**b. Microphone configurations:**

- “...the video source of AVA-AVD is an existing public video dataset...movies of different races, ample dialogues, and diverse languages and genres…”  
    _(No direct mention of microphone configuration; video-based dataset inherits original recording setup from AVA-Active Speaker.)_
    

---

## 3. **Speaker & Language Characteristics**

**a. Speakers per recording (average, range):**

- Table 1: “…number of clips vs. number of speakers in each clip; 2/7.7/24 [min/avg/max]…”
    

**b. Languages represented, distribution:**

- Table 1: “Multi(>6)”
    
- “...videos in multiple languages such as English, French, Chinese, German, Korean, Spanish, etc…”
    

---

## 4. **Annotations & Metadata**

**a. Annotation types provided:**

- “Diarization datasets only provide annotations in Rich Transcription Time Marked (rttm) format.”
    
- “...annotators need to refine the onset and offset of the speech segment proposals and get accurate segment labels.”
    
- “...labels for faces and active speakers…”
    

**b. Annotation format:**

- “diarization datasets only provide annotations in Rich Transcription Time Marked (rttm) format.”
    

**c. Forced-alignment or transcript segmentation:**

- _(No direct mention of forced-alignment; annotation focuses on segment timing and ID, not detailed transcripts.)_
    

---

## 5. **Licensing & Access**

**a. Licensing model:**

- “Our data and code has been made publicly available at [https://github.com/showlab/AVA-AVD.”](https://github.com/showlab/AVA-AVD.%E2%80%9D)
    
- “Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage…”
    

**b. Download links:**

- “Our data and code has been made publicly available at [https://github.com/showlab/AVA-AVD.”](https://github.com/showlab/AVA-AVD.%E2%80%9D)
    

---

## 6. **Data Splits & Benchmarks**

**a. Training/dev/test splits:**

- “We split 243 clips (20.25 hours) for training, 54 clips (4.5 hours) for validation, and 54 clips (4.5 hours) for testing.”
    
- Appendix: “...do not split the clips from the same movie into different subsets to keep the disjoint property across subsets…”
    

**b. Evaluation metrics recommended:**

- “The diarization performance is evaluated by Diarization Error Rate (DER), which is lower the better. It contains three terms: Missing Detection (MS), False Alarm (FA), and Speaker Error (SPKE).”
    

**c. Baseline results published:**

- Table 2, Table 3:  
    “Our AVRNet yields the best result 20.57 because it learns to compute the similarity of two audio-visual pairs...”  
    “...WST only achieves a DER of 42.04 on AVA-AVD...VBx has much higher SPKE (18.45) and DER (21.37) on AVA-AVD…”
    

---

## 7. **Usage & Preprocessing**

**a. Preprocessing steps suggested:**

- “For all of the experiments, we resize the input faces to 112 × 112. Following a standard preprocessing step , we run face alignment using RetinaFace .”
    
- “The spectrogram is extracted every 10ms with a window of 25ms on the fly.”
    

**b. Existing toolkits/scripts provided:**

- “Our data and code has been made publicly available at [https://github.com/showlab/AVA-AVD.”](https://github.com/showlab/AVA-AVD.%E2%80%9D)
    

---

## 8. **Challenges & Limitations**

**a. Known challenges:**

- “...challenging due to the diverse scenes, complicated acoustic conditions, and completely off-screen speakers…”
    
- “AVA-AVD has a large amount of background music, traffic noise, laughter, etc.”
    
- “Simple energy-based VAD systems cannot work well on AVA-AVD.”
    

**b. Limitations to consider:**

- “noted that diarization datasets only provide video-level speaker identities instead of global ones.”
    
- “Our system adopts unsupervised clustering and the number of speakers is unknown.”
    
- “...the original missing faces in AVA-AVD account for 50%.”
    

---

## 9. **Research Applications**

**a. Research tasks suited for dataset:**

- “audio-visual speaker diarization aims at detecting ‘who spoke when’ using both auditory and visual signals.”
    
- “translation, summarization, social analysis...speaker embedding, diarization, overlap detection…”
    

**b. Notable papers using this dataset:**

- “We build AVA-AVD dataset upon a publicly available dataset, i.e. AVA-Active Speaker… cited in multiple works: , , …”
    

---

## 10. **Reproducibility**

**a. Can the dataset be reproduced:**

- “Our data and code has been made publicly available at [https://github.com/showlab/AVA-AVD.”](https://github.com/showlab/AVA-AVD.%E2%80%9D)
    

**b. Version controls or update logs:**

- _(No explicit versioning or update logs mentioned in the paper; check GitHub repo for possible release details.)_
    

---

_All statements above are taken directly from quoted sections or tables in the AVA-AVD paper._

1. [https://arxiv.org/pdf/2111.14448.pdf](https://arxiv.org/pdf/2111.14448.pdf)