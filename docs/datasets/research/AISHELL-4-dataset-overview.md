**AISHELL-4**  
   Zhang, X., Shi, L., Wu, J., et al. “AISHELL-4: An Open-Source Dataset for Speech Separation and Recognition in Conference Scenario.” *Interspeech 2021*.  
   https://arxiv.org/abs/2104.03603  

---

**1. Dataset Overview**  
a. **Total Size**: 120 hours of real meeting recordings, consisting of 211 sessions (191 train, 20 eval, each ≈30 min).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
b. **Domains/Scenarios**: Real conference/meeting (Mandarin), recorded in small, medium, large conference rooms. Scenarios include medical, education, business, organizational, industrial, and daily routine meetings.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)

---

**2. Recording Conditions**  
a. **Audio Quality**: 8-channel circular microphone array, headset microphones; real-world conference acoustics. Sampling rates are not explicitly stated, but meeting scenarios include realistic noise (keyboard, door, fan), overlap, quick speaker turns.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
b. **Microphone Configurations**: 8-channel array (main), headset mics (near-field reference) per participant.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)

---

**3. Speaker & Language Characteristics**  
a. **Speakers per Recording**: Each session: 4–8 speakers; total: 36 (train), 25 (eval); gender-balanced.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
b. **Languages**: Mandarin (all speakers are native Mandarin speakers, no strong accents).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)

---

**4. Annotations & Metadata**  
a. **Annotation Types**:

- Speaker turns
    
- Overlap regions
    
- Speaker attribution & boundaries
    
- Transcriptions (with nonspeech events, e.g., laughter, cough)[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    b. **Annotation Format**: TextGrid format session files, includes segmentation, speaker info, timestamps, transcripts, overlap/non-overlap labels.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    c. **Forced Alignment/Transcripts**: Alignment via headset and array reference signal; ASR outputs used to assist labeling, Praat used for calibration.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
    

---

**5. Licensing & Access**  
a. **Licensing Model**: Open-source; available at [http://www.aishelltech.com/aishell](http://www.aishelltech.com/aishell).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
b. **Download Links**: Dataset home and baseline toolkit [https://github.com/felixfuyihui/AISHELL-4](https://github.com/felixfuyihui/AISHELL-4); ASR resources [https://github.com/usnistgov/SCTK](https://github.com/usnistgov/SCTK), SAD [http://kaldi-asr.org/models/12/0012](http://kaldi-asr.org/models/12/0012), speaker diarization [https://github.com/BUTSpeechFIT/VBx](https://github.com/BUTSpeechFIT/VBx).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
https://www.openslr.org/111/
https://github.com/felixfuyihui/AISHELL-4
---

**6. Data Splits & Benchmarks**  
a. **Train/Dev/Test Splits**: 107.5 h train (191 sessions), 12.72 h eval (20 sessions); evaluation sessions in distinct venues, with room overlap controlled.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
b. **Evaluation Metrics**:

- Character Error Rate (CER) for ASR (speaker-independent and dependent tasks)
    
- Diarization pipeline described, but DER not explicitly reported.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    c. **Baseline Results**:
    
- Speaker-independent ASR CER: 32.56% (no FE), 30.49% (with FE)
    
- Speaker-dependent ASR CER: 41.55% (no FE), 39.86% (with FE).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
    

---

**7. Usage & Preprocessing**  
a. **Preprocessing Steps**:

- VAD (via SAD; 5-layer TDNN)
    
- Speaker embedding extraction (ResNet on MFCC/filterbank features)
    
- SpecAugment for data augmentation.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    b. **Existing Toolkits/Scripts**:
    
- PyTorch-based baseline system (released)
    
- Kaldi recipes, VBx clustering toolkit, Praat calibration scripts.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
    

---

**8. Challenges & Limitations**  
a. **Known Challenges**:

- Real-world conference acoustics: noise, overlaps, speaker turns, non-grammatical speech
    
- Annotation complexity (multiple annotators, calibration required).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    b. **Limitations**:
    
- Mandarin only (limits cross-lingual comparison)
    
- No explicit mention of missing segments or license/data use restrictions beyond open-source.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
    

---

**9. Research Applications**  
a. **Research Tasks Supported**:

- Speaker diarization, overlap detection, speech enhancement/separation, ASR, speaker embedding, multi-modality modeling.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    b. **Notable Papers/Findings**:
    
- This is the primary publication; references other baseline systems/datasets (LibriCSS, CHiME-6, Switchboard, Fisher, AMI, ICSI, CHIL).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
    

---

**10. Reproducibility**  
a. **Reproducibility**:

- Scripts for training/evaluation released; details of data generation, calibration, and annotation process documented.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)  
    b. **Version Control/Update Logs**:
    
- Data hosted on GitHub; version history for code/toolkits.[arxiv](https://arxiv.org/pdf/2104.03603.pdf)
    

---

**Citation:**  
All details referenced from arXiv:2104.03603 (AISHELL-4 Dataset).[arxiv](https://arxiv.org/pdf/2104.03603.pdf)

1. [https://arxiv.org/pdf/2104.03603.pdf](https://arxiv.org/pdf/2104.03603.pdf)