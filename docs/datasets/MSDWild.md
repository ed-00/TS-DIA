 **MSDWild**  
   Liu, Y., Li, Z., Zhang, H., et al. “MSDWILD: Multi-Modal Speaker Diarization Dataset in the Wild.” *Interspeech 2022*.  
   https://www.isca-archive.org/interspeech_2022/liu22t_interspeech.pdf  

---

## 1. **Dataset Overview**

**a. Total Size**

- "**Our dataset, MSDWild, contains 3143 video clips with 84 labeled hours.**"
    
- Table 2: "Few-talker training set: 2476 clips, 69.09 hours; Few-talker val: 490 clips, 10.58 hours; Many-talker val: 177 clips, 4.51 hours."
    

**b. Domains & Scenarios**

- "**Daily casual conversation occupies the majority of our dataset. Compared with formal conversation: news report or debate, the daily casual conversation has three features: frequently talking in turn, various head gestures, and various background noises or room reverberations.**"
    

---

## 2. **Recording Conditions**

**a. Audio Quality**

- "**The sample rate of our audios is 16k Hz.**"
    
- Discusses casual videos: "various background noises or room reverberations."
    

**b. Microphone Configurations**

- Sourced from "public videos" (YouTube VLogs): "All video clips are naturally shot videos without over-editing such as lens switching."
    
- No mention of controlled microphone setups, as videos are "naturally shot."
    

---

## 3. **Speaker & Language Characteristics**

**a. Speakers per Recording**

- Table 2:
    
    - Few-talker train: "2 / 2.54 / 4"
        
    - Few-talker val: "2 / 2.61 / 4"
        
    - Many-talker val: "5 / 5.86 / 10"
        
- Quote: "**The min/average/max speaker number per video.**"
    

**b. Languages Represented**

- "**To improve the language diversity, we ... use Google Translate to translate those English keywords into different languages such as Chinese, Thai, Korean, Japanese, German, Portuguese, and Arabic.**"
    
- Table 1 includes the language column: "Multi" (multiple languages).
    

---

## 4. **Annotations & Metadata**

**a. Annotation Types**

- "**We mark different timelines for different speakers and add temporal segments for each speech duration. The opening and closing of the lips mark the beginning and end of a speech segment. Only speech is labeled while other human sounds such as laughing and singing are ignored.**"
    
- Baseline experiments: "We report two metrics: DER(Diarization Error Rate) and JER(Jaccard Error Rate)."
    

**b. Annotation Format**

- "**Manual labeling using VIA Video Annotator. VIA Video Annotator is a manual annotation software for videos, which has a video player and a timeline.**"
    

**c. Forced-Alignment / Transcript Segmentation**

- No explicit mention of forced alignment or transcript-level segmentation—labels are temporal speaker activity based on "opening and closing of the lips."
    

---

## 5. **Licensing & Access**

**a. Licensing Model**

- "**The dataset is available at [https://x-lance.github.io/MSDWILD](https://x-lance.github.io/MSDWILD).**"
    
- No explicit mention of license type; described as "fully released dataset that can be used for benchmarking..."
    

**b. Download Links**

- "_The dataset is available at [https://x-lance.github.io/MSDWILD](https://x-lance.github.io/MSDWILD)._"
    
- Not specified per type within the paper; presume audio & video available through this page.
    

---

## 6. **Data Splits & Benchmarks**

**a. Training/Dev/Test Splits**

- "**Our dataset forms three sets: few-talker training set, few-talker testing set, and many-talker testing set.**"
    
- Table 2 details: "2476 train / 490 val / 177 many-talker val"
    

**b. Evaluation Metrics**

- "**We report two metrics: DER(Diarization Error Rate) and JER(Jaccard Error Rate). DER ... summary of missed speech (MS) time, false alarm (FA) time, and speaker error (SE) time ... JER ... is the average of each speaker’s MS and FA rate.**"
    

**c. Baseline Results**

- Tables 3 and 4 present DER & JER scores for different baseline systems (audio-only, visual-only, audio-visual, two-stream, fused).
    

---

## 7. **Usage & Preprocessing**

**a. Preprocessing Steps**

- "**The visual data augmentation uses random horizontal flip, random cropping, random rotation, and random sampling low resolution: 32 × 32 , 64 × 64 , or 96 × 96. ... The audio data uses MUSAN and room reverberation for data augmentation.**"
    

**b. Toolkits or Scripts**

- "_Pyannote , a publicly available open-source toolkit^1 for speaker diarization,_"
    
- Footnote: "[https://github.com/pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)"
    

---

## 8. **Challenges & Limitations**

**a. Known Challenges**

- "**Speaker diarization in real-world acoustic environments is a challenging task... The many-talker set has much more speech alternations and overlapped speeches. The overall DER and JER show that our dataset is challenging, especially in the many-talker condition.**"
    
- Describes "domain mismatch, overlap ratio," compared to other datasets.
    

**b. Limitations**

- Not explicitly stated, but: "experiments reveal that there is still potential for improvement, particularly in the multi-talker situation."
    
- No mention of license restrictions or missing segments.
    

---

## 9. **Research Applications**

**a. Research Tasks**

- "**MSDWILD can provide a real in-the-wild testbed for the speaker diarization community. Besides, MSDWild is also suitable for exploring the ability of multi-modal audio-visual fusion for speaker diarization, better solving the problem of ‘who spoke when.’**"
    
- Sub-tasks: "speaker diarization, multi-modality, audio-visual, active speaker detection."
    

**b. Notable Papers & Findings**

- Baseline results reported in this paper; references to related works in section 2 and at the end.
    
- No other notable papers cited as using this dataset yet (dataset recently released).
    

---

## 10. **Reproducibility**

**a. Reproducibility**

- "**In this paper, we release MSDWild∗, a benchmark dataset for multimodal speaker diarization in the wild. ... These datasets can be used for training and testing simultaneously.**"
    

**b. Version Control / Update Logs**

- Not mentioned; main dataset site: "[https://x-lance.github.io/MSDWILD](https://x-lance.github.io/MSDWILD)" (check site for updates/logs).
    

---

**All information is sourced directly from the MSDWild paper as published on ISCA Archive.**

1. [https://www.isca-archive.org/interspeech_2022/liu22t_interspeech.pdf](https://www.isca-archive.org/interspeech_2022/liu22t_interspeech.pdf)