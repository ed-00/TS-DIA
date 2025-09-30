**Earnings-21**  
   Del Rio, E., Sainath, T. N., Shatilov, D., et al. “Earnings-21: A Practical Benchmark for ASR and Diarization in Financial Earnings Calls.” *Interspeech 2021*.  
   https://arxiv.org/abs/2104.11348  

---

## 1. **Dataset Overview**

**a. Total size (hours, segments)**

- “The Earnings-21 dataset consists of 44 public earnings calls recorded in 2020 from 9 corporate sectors ... totalling 39 hours and 15 minutes.”
    
- “The recordings in this corpus range in length from less than 17 minutes to 1 hour and 34 minutes with the average recording being about 54 minutes in length.”
    

**b. Domains and scenarios**

- All data consists of “earnings calls … downloaded from Seeking Alpha”, representing the domain of _financial sector teleconferences_.
    
- “Seeking Alpha defines 9 different sectors that categorize all earnings calls on their website: Basic Materials, Conglomerates, Consumer Goods, Financial, Healthcare, Industrial Goods, Services, Technology, and Utilities.”
    
- Scenario: “recordings created during the year 2020” featuring _real-world settings with diverse semantic and acoustic properties_.
    

---

## 2. **Recording Conditions**

**a. Audio quality (sampling rate, noise level)**

- “We chose recordings that had diverse sample rates … 44100 Hz, 24000 Hz, 22050 Hz, 16000 Hz, 11025 Hz.”
    
- “The audios are stored as monaural MP3 files.”
    
- “Audio quality: ... a wide range of recording qualities — representative of audio typically received in the wild ...”
    

**b. Microphone configurations**

- “We do not have any information on this audio metadata other than what can be inferred from the audios themselves. The audios are stored as monaural MP3 files.”
    

---

## 3. **Speaker & Language Characteristics**

**a. Speakers per recording**

- “The number of unique speakers ... is included in the speaker metadata.”
    
- “In some recordings, speakers are identified by name — when provided by the transcriptionists we include these in the speaker metadata.”
    

**b. Languages represented**

- “During transcription, a transcriber found that one of the earnings calls contained a large amount of non-English speech; we removed this call from the dataset ... because the remaining files still provide adequate coverage of entities over all sectors.”
    
- Therefore, all remaining files are English.
    

---

## 4. **Annotations & Metadata**

**a. Annotation types**

- “Entity-rich transcripts with annotated numerical figures.”
    
- “We assigned NER labels ... First we used our internal NER tools ... Next, we applied SpaCy 2.3.5’s NER tags ... Finally, we manually reviewed these tags and updated the entities.”
    
- “Richly annotated transcripts (with punctuation, true-casing, and named entities) for detailed error analysis.”
    
- “Speakers are identified by name — when provided by the transcriptionists we include these in the speaker metadata.”
    
- No explicit mention of “speaker turns, overlap labels, diarization error rates”.
    

**b. Annotation format**

- “At Rev, we store reference transcripts in a custom format file we call .nlp files. These files are .csv-inspired pipe-separated (i.e. ’|’) text files that present tokens and their metadata on separate lines.”
    
- Metadata: “file length in seconds, file sampling rate, the company name / sector, the calls financial quarter, the number of unique speakers and the total number of utterances in each recording.”
    

**c. Forced-alignment or transcript segmentation**

- “Using our recently released fstalign tool, we provide a candid analysis ... fstalign enables custom transformations ... useful for incorporating text normalization information to the WER calculation.”
    
- No explicit mention of forced-alignment output or segmentation files; segmentation present via token-level annotations.
    

---

## 5. **Licensing & Access**

**a. Licensing model**

- “Earnings-21, an open and free evaluation corpus ... available on Github ... Earnings calls fair use legal precedent in Swatch Group Management Services Ltd. v. Bloomberg L.P.”
    

**b. Download links**

- “The dataset is available on Github: [https://github.com/revdotcom/speechdatasets/tree/master/earnings21”](https://github.com/revdotcom/speechdatasets/tree/master/earnings21%E2%80%9D)
    
- “fstalign: [https://github.com/revdotcom/fstalign”](https://github.com/revdotcom/fstalign%E2%80%9D)
    

---

## 6. **Data Splits & Benchmarks**

**a. Training/dev/test splits**

- No explicit mention of train/dev/test splits; the dataset is for _evaluation_.
    

**b. Evaluation metrics**

- “We provide detailed WER analysis ...”
    
- “fstalign ... for quickly computing WER that leverages NER annotations.”
    
- Main evaluation metric: Word Error Rate (WER) and entity-level WER.
    

**c. Baseline results published**

- “Table 2: Comparison of WER overall models on the Earnings-21 dataset.”
    
- Systems benchmarked: Google, Amazon, Microsoft, Speechmatics, Rev, Kaldi.org, Kaldi ESPNet, LibriSpeech.
    
- “We present the results of our WER measurements in Table 2. We find that the ESPNet model is the most accurate on Earnings-21.”
    

---

## 7. **Usage & Preprocessing**

**a. Preprocessing steps**

- “Audio is resampled at 16KHz for training and inference time” (for internal models).
    
- Text normalization: “We assigned NER labels ... tagged tokens that require text normalization such as abbreviations, cardinals, ordinals, and contractions.”
    

**b. Toolkits/scripts for data**

- “fstalign, a novel toolkit for quickly computing WER that leverages NER annotations.”
    
- “Our models were developed as part of general ASR systems with training data sourced from an unbiased selection from our database. ... Kaldi ... ESPNet ...”
    

---

## 8. **Challenges & Limitations**

**a. Known challenges**

- “Entity-rich benchmarks ... ASR accuracy for certain NER categories is poor ... significant impediment to transcript comprehension and usage.”
    
- “We have intentionally chosen to keep this difficult file as it presents realistic lens into the variability of audio in the wild.”
    
- “Major obstacles to speech recognition in the wild.”
    

**b. Limitations before training**

- “We do not have any information on this audio metadata other than what can be inferred from the audios themselves.”
    
- “During transcription, ... non-English speech ... removed.”
    
- “Many of these traditional evaluation sets are not free to use, limiting access ... The most challenging public test suite our team has used ...”
    

---

## 9. **Research Applications**

**a. Research tasks**

- “enables further research on entity modeling and WER on real world audio.”
    
- “We challenge researchers to deal with real-world audio.”
    
- Focus areas: ASR, Named Entity Recognition, speech in challenging domains.
    

**b. Notable papers using dataset**

- None specified beyond this paper.
    

---

## 10. **Reproducibility**

**a. Can the dataset be reproduced?**

- “The commercial model output is provided in the data release for convenient reproducibility.”
    
- “We provide fstalign as a tool to enable the research community ...”
    

**b. Version controls / update logs**

- Not explicitly mentioned; updates can be tracked on the Github repository.
    

---

If you need full quotes or tables for any specific section, let me know!

1. [https://arxiv.org/pdf/2104.11348.pdf](https://arxiv.org/pdf/2104.11348.pdf)