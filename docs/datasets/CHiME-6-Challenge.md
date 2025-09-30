**CHiME-6**  
    Watanabe, S., et al. “The 6th CHiME Challenge: Multispeaker Speech in Home Environments.” *arXiv:2004.09249*, 2020.  
    https://chimechallenge.github.io/chime6/  

---
## 1. **Dataset Overview**

- **Total size (hours, segments):**  
    “The data have been split into training, development test, and evaluation test sets as follows.  
    
    | Dataset | Parties | Speakers | Hours | Utterances |  
    | Train | 16 | 32 | 40:33 | 79,980 |  
    | Dev | 2 | 8 | 4:27 | 7,440 |  
    | Eval | 2 | 8 | 5:12 | 11,028 |”
    
- **Domains and scenarios:**  
    “Speech material has been collected from twenty real dinner parties that have taken place in real homes. … The parties have been made using multiple 4-channel microphone arrays and have been fully transcribed. … Each party should last a minimum of 2 hours and should be composed of three phases, each corresponding to a different location:
    
    - kitchen – preparing the meal in the kitchen area;
        
    - dining – eating the meal in the dining area;
        
    - living – a post-dinner period in a separate living room area.”
        

---

## 2. **Recording Conditions**

- **Audio quality:**  
    “All audio data are distributed as WAV files with a sampling rate of 16 kHz. … real domestic noise backgrounds, e.g., kitchen appliances, air conditioning, movement, etc.”
    
- **Microphone configurations:**  
    “Each party has been recorded with a set of six Microsoft Kinect devices. … Each Kinect device has a linear array of 4 sample-synchronised microphones and a camera. … In addition to the Kinects, to facilitate transcription, each participant is wearing a set of Soundman OKM II Classic Studio binaural microphones.”  
    “Each session consists of the recordings made by the binaural microphones worn by each participant (4 participants per session), and by 6 microphone arrays with 4 microphones each. Therefore, the total number of microphones per session is 32 (2 x 4 + 4 x 6).”
    

---

## 3. **Speaker & Language Characteristics**

- **Speakers per recording (average, range):**  
    “Each dinner party has four participants - two acting as hosts and two as guests.”  
    “Speakers: Train 32, Dev 8, Eval 8. (4 speakers per session.)”
    
- **Languages represented:**  
    (Not explicitly listed. The official description implies “natural conversational speech” in “real homes;” prior challenge editions primarily used English.)
    

---

## 4. **Annotations & Metadata**

- **Annotation types provided:**  
    “Fully transcribed utterances are provided in continuous audio with ground truth speaker labels and start/end time annotations for segmentation.”  
    “The JSON file includes … speaker ID, transcription, start time, end time, reference microphone array ID, location, session ID.”
    
- **Annotation format:**  
    “The transcriptions are provided in JSON format for each session as .json.”
    
- **Forced-alignment or transcript segmentation:**  
    “The CHiME-6 software provided will modify the signals and transcripts to generate an improved alignment, in particular, compensating for frame-drops and clock-skew. These modified signals form the starting point for CHiME-6.”
    

---

## 5. **Licensing & Access**

- **Licensing model:**  
    “The CHiME-6 challenge requires use of the CHiME-5 dataset. This dataset is available under licence.  
    Both commercial and non-commercial licences are available.”
    
- **Download links:**  
    “All data is available for [download](https://chimechallenge.github.io/chime6/download.html) under licence.”
    

---

## 6. **Data Splits & Benchmarks**

- **Training/dev/test splits:**  
    “The data have been split into training, development test, and evaluation test sets as follows.” (See above)
    
- **Evaluation metrics:**  
    “the impact of diarization error on speech recognition error will be measured,”  
    “state-of-the-art baselines are provided for diarization, enhancement, and recognition.”
    
- **Baseline results published:**  
    “The tables below present results for both challenge tracks …” (Lists DER scores and links to system papers.)
    

---

## 7. **Usage & Preprocessing**

- **Preprocessing steps suggested:**  
    “The CHiME-6 software provided will modify the signals and transcripts to generate an improved alignment, in particular, compensating for frame-drops and clock-skew.”
    
- **Existing toolkits or scripts:**  
    “an accurate array synchronization script is provided,”  
    “baseline recognition and evaluation tools,”  
    “All data is available for download under licence.”
    

---

## 8. **Challenges & Limitations**

- **Known challenges:**  
    “real domestic noise backgrounds, e.g., kitchen appliances, air conditioning, movement, etc."  
    "a range of room acoustics from 20 different homes …”  
    “Notes” field for data sessions includes details such as “mic unreliable,” “missing,” “accidentally turned off,” “crashed,” “Neighbour interrupts,” “disconnects for bathroom.”
    
- **Limitations:**  
    “The binaural microphone recordings for the evaluation set can be used for array sychronization only. They shall not be used for diarization, enhancement, and recognition.”  
    “Some personally identifying material has been redacted post-recording …”  
    “Possession of a valid licence will be a pre-requisite for submission to the challenge.”
    

---

## 9. **Research Applications**

- **Suitable research tasks:**  
    “The challenge moves beyond automatic speech recognition (ASR) and also considers the task of diarization, i.e., estimating the start and end times and the speaker label of each utterance.”
    
- **Notable papers using the dataset:**  
    “To refer to these data in a publication, please cite: Jon Barker, Shinji Watanabe, Emmanuel Vincent, and Jan Trmal The fifth ‘CHiME’ Speech Separation and Recognition Challenge: Dataset, task and baselines Interspeech, 2018.”
    

---

## 10. **Reproducibility**

- **Can the dataset be reproduced:**  
    “an accurate array synchronization script is provided,”  
    “CHiME-6 software provided will modify the signals and transcripts to generate an improved alignment …”
    
- **Version controls or update logs:**  
    (No explicit mention found on website.)
    

---

If you want full technical documentation or sample downloads, go to the [Download page](https://chimechallenge.github.io/chime6/download.html) and request a dataset licence as described above.

1. [https://chimechallenge.github.io/chime6/](https://chimechallenge.github.io/chime6/)
2. https://openslr.org/150/