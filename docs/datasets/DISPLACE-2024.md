**DISPLACE 2024**  
    Durmuş, E., et al. “The Second DISPLACE Challenge: DIarization in Conversational Environments.” *arXiv:2406.09494*, 2024.  
    https://displace2024.github.io  

---

## 1. **Dataset Overview**

**a. Total size (hours, segments):**

- "The DISPLACE corpus comprises roughly 32 hours of conversational data."
    

**b. Domains and scenarios:**

- "The DISPLACE corpus is a collection of informal conversations ... involving a group of 3 to 5 participants."
    
- "The goal is to perform speaker diarization (who spoke when) in multilingual conversational audio data, where the same speaker speaks in multiple code-mixed and/or code switched languages."
    
- Scenarios: "climate change, culture, politics, sports, entertainment, and more."
    
- "There will be no training data given and the participants will be free to use any resource for training the models."
    

---

## 2. **Recording Conditions**

**a. Audio quality (sampling rate, noise level):**

- "The recordings are single-channel wav files sampled at 16 kHz. Close-talking recordings were time-synchronized with far-field audio and normalized to [-1, 1] amplitude range."
    
- "Noise cancellation enabled" (for Android/ASR app recordings).
    

**b. Microphone configurations:**

- "Each participant was given a lapel microphone to wear, while a shared far-field desktop microphone was positioned at the center of the table."
    
- "Far-field recordings at the IISc were conducted using an omnidirectional microphone ... In contrast ... at NITK, a unidirectional far-field microphone ... All recordings consist of single-channel data."
    

---

## 3. **Speaker & Language Characteristics**

**a. Speakers per recording:**

- "each conversation lasting between 30 to 60 minutes and involving a group of 3 to 5 participants."
    

**b. Languages represented & distribution:**

- "Participants ... required to have proficiency in at least one Indian language (L1) and English, with an Indian accent. ... collectively chosen ..."
    
- Zenodo lists: "Languages: English, Kannada, Telugu, Hindi, Bengali"
    

---

## 4. **Annotations & Metadata**

**a. Annotation types provided:**

- "We will require you to submit the output decisions as a Rich Transcription Time Marked (RTTM) file for system evaluation."
    
- "for annotation purposes (for ease of human listening to determine the speech, speaker and language boundaries)."
    

**b. Annotation format:**

- "Rich Transcription Time Marked (RTTM) file."[displace2024.github](https://displace2024.github.io/index.html)
    
- File naming conventions: "A unique file ID is assigned ... Close-talking Mic1: 20221011 1545 B001 S1 En.wav ... Far-field audio is recorded ... Spoken Languages: ... e.g., 'MaHiEn'."
    

**c. Forced-alignment or transcript segmentation:**

- "Close-talking recordings were time-aligned with far-field audio, resampled to 16 kHz, and normalized ... The worn microphone speech was recorded for annotation purposes."
    

---

## 5. **Licensing & Access**

**a. Licensing model:**

- "Creative Commons Attribution 4.0 International"
    
- "You can use it for research purposes with proper citations ... No, you can not re-distribute the data even if you have participated in the challenge."[displace2024.github](https://displace2024.github.io/index.html)
    

**b. Download links for audio, annotations, documentation:**

- Dataset: [https://zenodo.org/records/12166687](https://zenodo.org/records/12166687) (Password Protected)
    
- Baseline Systems: [https://github.com/displace2024/Displace2024_baseline_updated](https://github.com/displace2024/Displace2024_baseline_updated)
    
- Evaluation Plan & flyer: [https://displace2024.github.io/docs/DISPLACE2024_Evaluation_Plan_v2.pdf](https://displace2024.github.io/docs/DISPLACE2024_Evaluation_Plan_v2.pdf), [https://displace2024.github.io/docs/DISPLACE2024_Flyer_v5.pdf](https://displace2024.github.io/docs/DISPLACE2024_Flyer_v5.pdf)[displace2024.github](https://displace2024.github.io/index.html)
    

---

## 6. **Data Splits & Benchmarks**

**a. Training/dev/test splits:**

- "You will be provided with a dev set (far-field recordings), and a baseline system ... Subsequently, a blind evaluation set (far-field recordings), will be provided ... submit your model predictions ... on the blind set to a leaderboard interface (setup in Codalab)."[displace2024.github](https://displace2024.github.io/index.html)
    

**b. Evaluation metrics recommended:**

- "The performance metric for evaluation will be the Diarization Error Rate (DER)."[displace2024.github](https://displace2024.github.io/index.html)
    
- "For track 3 ... overall evaluation ... in terms of Word Error Rate (WER)."[displace2024.github](https://displace2024.github.io/index.html)
    
- "DER, WER"[displace2024.github](https://displace2024.github.io/index.html)
    

**c. Baseline results published:**

- "We computed the Word Error Rate (WER) for close field recording per language ..."
    

text

`| Language | WER (Dev) | | Hindi | 58.5 | | Bengali | 63.5 | | Telugu | 71.2 | | Kannada | 80.8 | | English from all sessions | 66.5 | | Overall | 66.7 |`

- "Baseline for speaker diarization ... Pyannote SAD model ... Spectral Clustering ... VB-HMM resegmentation ... Pyannote overlap detector ... DER."
    

---

## 7. **Usage & Preprocessing**

**a. Preprocessing steps suggested:**

- "Speech activity detection using [Pyannote SAD model] ... Overlap handling using [Pyannote overlap detector] and VB-HMM together in the final stage."
    
- "Lapels recorded for annotation purposes ... time-aligned ... normalized ... speech activity detection ..."
    

**b. Existing toolkits/scripts provided:**

- "Baseline Systems: [Click here](https://github.com/displace2024/Displace2024_baseline_updated)"
    
- "Recipes from a fresh virtual environment ... Kaldi and dscore ... scripts in the `tools/` directory."
    
- Requirements and instructions for installing dependencies, running baselines, and using Kaldi/dscore.
    

---

## 8. **Challenges & Limitations**

**a. Known challenges:**

- "No publicly available dataset matches the diverse characteristics ... code-mixing/switching, natural overlaps, reverberation, and noise."
    
- "The current speaker diarization systems are simply not equipped to deal with multilingual conversations, where the same talker speaks in multiple code-mixed languages."[displace2024.github](https://displace2024.github.io/index.html)
    

**b. Limitations to consider:**

- "You can not re-distribute the data ... use it for research purposes with proper citations."
    
- "There will be no training data given ..."
    
- "Missing files in Speaker Diarization baseline have been updated."
    
- "Some segments are time-aligned for annotation, not all for diarization use."
    

---

## 9. **Research Applications**

**a. Suited research tasks:**

- "establish new benchmarks for speaker diarization (SD) in multilingual settings, language diarization (LD) in multi-speaker settings, and ASR in multi-accent settings, using the same underlying dataset."[displace2024.github](https://displace2024.github.io/index.html)
    
- "Speaker embedding, diarization, overlap detection, ASR, language diarization ..."[displace2024.github](https://displace2024.github.io/index.html)
    

**b. Notable papers using this dataset:**

- "The Second DISPLACE Challenge: DIarization of SPeaker and LAnguage in Conversational Environments." [https://arxiv.org/abs/2406.09494](https://arxiv.org/abs/2406.09494); Interspeech 2024 accepted papers[displace2024.github](https://displace2024.github.io/index.html)
    
- "Summary of the DISPLACE Challenge 2023 ..." [https://doi.org/10.1016/j.specom.2024.103080](https://doi.org/10.1016/j.specom.2024.103080)[displace2024.github](https://displace2024.github.io/index.html)
    

---

## 10. **Reproducibility**

**a. Reproducible:**

- "Baseline systems ... scripts in the tools/ directory ... instructions in README.md ..."
    

**b. Version control/update logs:**

- "Updates: [01/07/2024]: The Second DISPLACE challenge ([paper](http://www.arxiv.org/pdf/2406.09494)). ... [20/02/2024]: Missing files updated. ... [8/02/2024]: Track 3 (ASR) baseline details and results on DEV data updated."
    

---

**Download/access links:**

- **Dataset**: [Zenodo: DISPLACE-2024 Dataset (Password Protected)](https://zenodo.org/records/12166687)
    
- **Registration for access:** [Google Form](https://forms.gle/PcewWwYtVDwjZwg99)
    
- **Baseline systems/scripts:** [GitHub](https://github.com/displace2024/Displace2024_baseline_updated)
    
- **Documentation**: [Evaluation Plan](https://displace2024.github.io/docs/DISPLACE2024_Evaluation_Plan_v2.pdf)[displace2024.github](https://displace2024.github.io/index.html)
    
- **Flyer & overview**: [Flyer PDF](https://displace2024.github.io/docs/DISPLACE2024_Flyer_v5.pdf)[displace2024.github](https://displace2024.github.io/index.html)
    

---

**References:**  
DISPLACE Challenge main website[displace2024.github](https://displace2024.github.io/index.html)  
Corpus Overview & Data Sheet  
Zenodo dataset and access details  
GitHub baseline systems & documentation

1. [https://displace2024.github.io/index.html](https://displace2024.github.io/index.html)