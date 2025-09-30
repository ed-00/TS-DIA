**Synthetic LibriSpeech Corpus**  
   Suendermann, D., et al. “A Free Synthetic Corpus for Speaker Diarization Research.” *SPECOM 2018*.  
   http://suendermann.com/su/pdf/specom2018d.pdf  

---

## 1. **Dataset Overview**

- **Total size (hours, segments):**
    
    - "It includes over 90 hours of training data, and over 9 hours each of development and test data. Both 2-person and 3-person dialogs, with and without overlap, are included."
        
    - See Tables 1-4 for segment counts, e.g. for 2-person no-overlap training: "Train 292 [dialogs] 28522 [turns] 989715 [tokens] 98.15 [hours]" and for 3-person overlap training: "Train 195 26694 928346 90.64" [hours].[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Domains and scenarios:**
    
    - "A synthetic corpus of dialogs was made from the open-source LibriSpeech corpus... sections of English audio books recorded at 16 kHz sample rate"
        
    - "2-person and 3-person dialogs"
        
    - _Scenario_ is **simulated dialog** (not meetings, telephone, broadcast, or egocentric; based on clean audiobooks).[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 2. **Recording Conditions**

- **Audio quality (sampling rate, noise):**
    
    - "recorded at 16 kHz sample rate , usually with clear articulation and high-quality audio"
        
    - "very little background noise (but users could add their own for a better approximation to real conditions)".[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Microphone configurations:**
    
    - Not explicitly stated; original LibriSpeech is "high-quality readings of audio books" (single channel per speaker).[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 3. **Speaker & Language Characteristics**

- **Speakers per recording (average, range):**
    
    - "Both 2-person and 3-person dialogs, with and without overlap, are included."
        
    - 2-speaker dialogs: "Dialogs ranged in duration from 2.7-49.6 min (median 17.5 min)"
        
    - 3-speaker dialogs: "Dialogs included between 17 and 366 utterances (median 118), and ranged in duration from 2.8-71.5 min (median 24.4 min)."
        
    - "Each chapter has up to 129 utterances"—dialog construction described in detail.[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Languages represented:**
    
    - "The LibriSpeech corpus consists of sections of English audio books..."
        
    - Only **English**.[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 4. **Annotations & Metadata**

- **Annotation types:**
    
    - "Timing information is provided in 3 formats: 1) the Kaldi .ctm format; 2) the NIST .rttm format , as required by the widely-used md-eval-v21.pl script for computing the diarization error rate (DER) ; and 3) a simple frame-by-frame list of integer labels. In the later, 0 indicates silence, 1 indicates speaker 1, and 2 indicates speaker 2, etc. Integers greater than 10 indicate overlap.".[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
    - "includes not only speaker segmentations, but also phoneme segmentations."
        
    - _No explicit mention_ of overlap labels or diarization error rates in the annotation files, but format supports these.
        
- **Annotation format:**
    
    - ".ctm, .rttm, frame-by-frame list".[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Forced-alignment/transcript segmentation:**
    
    - "The dialog .ctm files include the timing information for individual phonemes, as obtained by forced alignment ... using the Kaldi 'ali2phones' utility".[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 5. **Licensing & Access**

- **Licensing model:**
    
    - "freely available for download at: [https://github.com/EMRAI/emrai-synthetic-diarization-corpus."[1](https://github.com/EMRAI/emrai-synthetic-diarization-corpus.%22%5B1)]
        
    - "open-source LibriSpeech corpus" is referenced as source material.
        
- **Download links:**
    
    - "[https://github.com/EMRAI/emrai-synthetic-diarization-corpus".[1](https://github.com/EMRAI/emrai-synthetic-diarization-corpus%22.%5B1)]
        

---

## 6. **Data Splits & Benchmarks**

- **Training/dev/test splits:**
    
    - "For training data, we use the 'train clean 100' subset ... The LibriSpeech 'dev clean', 'dev other', 'test clean', and 'test other' sets were likewise prepared for diarization development and test sets (Table 1)."[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Evaluation metrics:**
    
    - ".rttm format ... widely-used md-eval-v21.pl script for computing the diarization error rate (DER) "[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
    - **DER** is recommended; no mention of JER or CDER.
        
- **Baseline results:**
    
    - No baseline DER scores or system descriptions presented in this paper; corpus intended for system development.[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 7. **Usage & Preprocessing**

- **Preprocessing steps suggested:**
    
    - No explicit recommendations; "very little background noise (but users could add their own for a better approximation to real conditions )".[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Toolkits/scripts for loading:**
    
    - "The open-source and widely-used Kaldi speech recognition toolkit includes a recipe for ASR training and alignment of the LibriSpeech corpus."[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 8. **Challenges & Limitations**

- **Known challenges:**
    
    - "As a synthetic corpus, there are several deviations from real-world data. First, there is very little background noise ... Second, conversational statistics were approximately mimicked, but cannot be considered perfectly realistic. Third, we included no intervals of truly multi-speaker speech, i.e., 'back-channel' utterances by one speaker that occur fully within the turn of another speaker. Fourth, the LibriSpeech corpus itself consists of high-quality readings of audio books ... makes the speech unrealistic to most real-world applications. Fifth, ... no child or other special categories of speech. Finally, ... only include 2-speaker and 3-speaker dialogs (and 4-speaker dialogs will be included in a future release)."[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Limitations for training:**
    
    - "NOT suggest that the synthetic corpus replaces the need for real-world data; applied workers must also obtain data for each particular application."[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 9. **Research Applications**

- **Research tasks suited for:**
    
    - "diarization system development," "phoneme specificity," "phone adaptive training," "forced alignment," "speaker segmentation," "speaker embedding" (methodology description).[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Notable papers using dataset:**
    
    - Companion and reference papers are cited, but not detailed usage of this corpus.[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

## 10. **Reproducibility**

- **Can be reproduced (scripts/data generation):**
    
    - "Timing information is provided in several formats, and includes not only speaker segmentations, but also phoneme segmentations."
        
    - Corpus constructed from open-source LibriSpeech; methodology detailed for reproduction.[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        
- **Version control/update logs:**
    
    - Updates (e.g., "4-speaker dialogs will be included in a future release"), but no explicit mention of version logs.[suendermann](https://suendermann.com/su/pdf/specom2018d.pdf)
        

---

**All above content and quotes are directly from:**  
Paper: "A Free Synthetic Corpus for Speaker Diarization" by Edwards et al. ([https://suendermann.com/su/pdf/specom2018d.pdf).[1](https://suendermann.com/su/pdf/specom2018d.pdf\).%5B1)]

1. [https://suendermann.com/su/pdf/specom2018d.pdf](https://suendermann.com/su/pdf/specom2018d.pdf)