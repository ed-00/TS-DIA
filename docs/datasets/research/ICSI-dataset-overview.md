
1. **Dataset Overview**
    
    - **Total size (hours, segments):**  
        "The corpus as it will be released contains 75 meetings, for a total of about 72 hours. The meetings average about 6 participants per meeting, and each meeting also includes the audio from the 6 table-top microphones."
        
    - **Domains and scenarios:**  
        "Most of the recordings are of regular group meetings of research groups at ICSI."  
        _Meeting types:_
        
        - Even Deeper Understanding (Bed): "participants discuss natural language processing and neural theories of language."
            
        - Meeting Recorder (Bmr): "concerned with the ICs1 Meeting Corpus."
            
        - Robusmess (Bro): "methods to compensate for noise, reverberations, and other environmental issues in speech recognition."
            
        - Network Services &Applications (Bns): "researches internet architectures, standards, and related networking issues."
            
        - "The remaining recordings include meetings among the transcriptionists of the corpus, site visits from collaborators, and miscellaneous other meetings."
            
        - **No explicit telephone or broadcast scenarios** are described.
            
2. **Recording Conditions**
    
    - **Audio quality (sampling rate, noise level):**  
        "The data were down-sampled on the fly from 48 kHz to 16 kHz, and encoded using 16 bit linear NIST SPHERE format... higher quality settings do not appear to be necessary for automatic speech recognition systems."  
        "A projection screen is located at the end of the room. Although the projector was seldom used, its fan was active during the recordings."
        
    - **Microphone configurations:**  
        "For each meeting, we simultaneously recorded up to 10 closetalking head-worn microphones, 4 desktop omni-directional PZM microphones, and a 'dummy' PDA containing two inexpensive microphones... A few of the earlier meetings also used a single lapel-style microphone..."
        
3. **Speaker & Language Characteristics**
    
    - **Speakers per recording:**  
        "The meetings average about 6 participants per meeting... 53 Unique speakers 13 Female 40 Male."
        
    - **Languages represented, distribution:**  
        "36 American [English], 6 British, 2 Indian, 8 Unspecified" (Variety of English).  
        Native languages: "28 English, 12 German, 5 Spanish, 1 Chinese, 1 Czech, 1 Dutch, 1 French, 1 Hebrew, 1 Malayalam, 1 Norwegian, 1 Turkish - 5 Unspecified."
        
4. **Annotations & Metadata**
    
    - **Annotation types:**  
        "For each meeting, the corpus contains an XML file with a wordlevel transcription. In addition to the full words, other information is also provided, such as word fragments, restarts, filled pauses, back-channels, contextual comments (e.g. “while whispering”), and nan-lexical events such as cough, laugh, breath, lip smack, door slam, microphone clicks, etc."  
        "Overlap between participant’s speech is extremely common... In the transcript, we mark the speaker, the stan time, and the end time of each of the utterances."
        
    - **Annotation format:**  
        "The XML transcription format was designed specifically for this collection. A complete DTD and description of the format will be distributed with the corpus. We will also provide software for translating from our format to other common formats."
        
    - **Forced alignment/transcript segmentation:**  
        "...Each of the near-field signals was transcribed separately, and went through several passes of transcription, correction, and quality assurance... The linearized audio was sent to a commercial transcription service. Upon return, we divided the audio back into a separate channel for each speaker... corrected... using a version of the Transcriber tool modified for multiple channels. Finally, a senior transcriptionist verified the data."
        
5. **Licensing & Access**
    
    - **Licensing model:**  
        "We will deliver the corpus to the LDC [I] by December, 2002, and expect it to be available through the LDC by the summer of 2003." (LDC license required for access)
        
    - **Download links:**  
        Direct download links are not provided; see:
        
        - “Linguistic data consortium (LDC) web page,” [http://www.ldc.upenn.edu/](http://www.ldc.upenn.edu/)
            
        - “ICSI meeting corpus web page,” [http://www.icsi.berkeley.edu/speech/mr](http://www.icsi.berkeley.edu/speech/mr)
            
6. **Data Splits & Benchmarks**
    
    - **Training/dev/test splits:**  
        Not explicitly defined in this document; corpus includes all meetings.
        
    - **Evaluation metrics:**  
        "Overlap analysis, applications, prosody, automatic punctuation, noise robustness, and reverberation..."  
        No explicit mention of DER/JER/CDER in this document, but these are common for diarization.
        
    - **Baseline results/system descriptions:**  
        "We have published many papers related to the corpus, including research on automatic transcription, speech activity detection for segmentation, overlap analysis, applications, prosody, automatic punctuation, noise robustness, and reverberation... For an overview, please see ... For a complete listing of publications from ICSI on the Meeting Corpus, see our web page."
        
7. **Usage & Preprocessing**
    
    - **Preprocessing steps suggested:**  
        "Speech activity detection... The process of transcribing the data was quite complex. Each of the near-field signals was transcribed separately, and went through several passes... speech-activity detector..."
        
    - **Existing toolkits/scripts:**  
        "We will also provide software for translating from our format to other common formats."
        
8. **Challenges & Limitations**
    
    - **Known challenges:**  
        "Overlap between participant’s speech is extremely common in our meetings.... sections of a meeting that participants want excluded from public release... replaced with a pure tone on all channels..."  
        "Background noise and crosstalk... Problems with background noise and crosstalk (hearing a neighboring voice on the lapel's channel)..."
        
    - **Limitations:**  
        "Because of privacy concerns, the participant approval forms are not part of the released corpus... sections of meetings excluded by participant request."
        
9. **Research Applications**
    
    - **Research tasks:**  
        "Such a corpus supports work in automatic speech recognition, noise robustness, dialog modeling, prosody, rich transcription, information retrieval, and more."
        
    - **Notable papers:**  
        "We have published many papers related to the corpus, including research on automatic transcription, speech activity detection for segmentation, overlap analysis, applications, prosody, automatic punctuation, noise robustness, and reverberation... For an overview, please see ... For a complete listing of publications from ICSI on the Meeting Corpus, see our web page."
        
10. **Reproducibility**
    
    - **Can the dataset be reproduced?:**  
        "We will also provide software for translating from our format to other common formats."  
        Additional annotation and versions are ongoing: "we also continue to annotate the corpus with additional information, including dialog act labeling and prosodic features."
        
    - **Version controls/update logs:**  
        Ongoing annotation work noted; see the ICSI meeting corpus web page for updates.
        

**All information directly quoted or paraphrased from the provided document text as required.**

1. [https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1198793](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1198793)
