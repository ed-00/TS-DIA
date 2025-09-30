**CSSD**  
    Han, B., et al. “The Conversational Short-phrase Speaker Diarization (CSSD) Challenge.” *arXiv:2208.08042*, 2022.  
    https://arxiv.org/abs/2208.08042  

---

**1. Dataset Overview**

a. **Total size (hours, segments)**  
"MagicData-RAMC contains 180 hours of dialog speech in total. The dataset is divided into 149.65 hours training set, 9.89 hours development set, and 20.64 hours test set ... consisting of 289, 19, and 43 conversations, respectively. ... Each conversation is of 30.80 minutes duration on average. ... Table 2: Statistics ... #Segments Per Sample 1215 231 624.86"arxiv+1

b. **Domains and scenarios**  
"The dataset is collected indoors. ... The environments are relatively quiet ... Topics are freely chosen by the participants and the conversations in MagicData-RAMC are classified into 15 diversified domains, ranging from science and technology to ordinary life."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

**2. Recording Conditions**

a. **Audio quality**  
"All recording devices work at 16 kHz, 16-bit to guarantee high recording quality. ... ambient noise level lower than 40 dB."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Microphone configurations**  
"The audios are recorded over mainstream smartphones, including Android phones and iPhones. The ratio of Android phones to iPhones is around 1:1."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

**3. Speaker & Language Characteristics**

a. **Speakers per recording**  
"There are a total of 663 speakers involved ... Each speaker participates in up to three conversations. ... The number of participants involved in three subsets are 556, 38, and 86, respectively."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Languages and distribution**  
"All participants are native and fluent Mandarin Chinese speakers with slight variations of accent ... accent region is roughly balanced, with 334 Northern Chinese speakers and 329 Southern Chinese speakers."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

**4. Annotations & Metadata**

a. **Annotation types provided**  
"Each segment is labeled with the corresponding speaker-id. ... Sound segments without semantic information ... are annotated with specific symbols. ... The precise voice activity timestamps of each speaker are provided."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Annotation format**  
"The original partition of the speech data is provided in TSV format ... All transcriptions ... are prepared in TXT format for each dialog."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

c. **Forced-alignment/transcript segmentation**  
"Sound segments ... are annotated with specific symbols ... Disfluencies ... are recorded and fully transcribed. ... The start and end times of all segments are specified to within a few milliseconds."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

**5. Licensing & Access**

a. **Licensing model**  
"We have released the open-source MagicData-RAMC ... prepared an individual 20-hour conversational speech test dataset with artificially verified speakers timestamps annotations for the CSSD task."openslr+1

b. **Download links**  
"MagicData-RAMC: [https://magichub.com/datasets/magicdata-ramc](https://magichub.com/datasets/magicdata-ramc)"  
"CSSD metric and baseline: [https://github.com/SpeechClub/CDER_Metric](https://github.com/SpeechClub/CDER_Metric)"  
"Baseline system: [https://github.com/MagicHub-io/MagicData-RAMC"[1](https://github.com/MagicHub-io/MagicData-RAMC%22%5B1)]

---

**6. Data Splits & Benchmarks**

a. **Training/dev/test splits**  
"149.65 hours training set, 9.89 hours development set, and 20.64 hours test set ... Table 1 ..."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Evaluation metrics recommended**  
"Traditionally, diarization error rate (DER) has been used ... we design the new conversational DER (CDER) evaluation metric, which calculates the SD accuracy at the utterance level."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

c. **Baseline results published**  
"Table 3: Speaker diarization results of VBx system ... DER Dev 5.57, Test 7.96 ... CDER Dev 17.48, Test 19.90 (0.25s collar); CDER Dev 26.9, Test 28.2 (0s collar)"[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

**7. Usage & Preprocessing**

a. **Preprocessing steps suggested**  
"For training details, the SAD module utilizes 40-dimensional Mel frequency cepstral coefficients (MFCC) with 25 ms frame length and 10 ms stride ... Speaker embeddings are extracted on SAD result every 240 ms, and the chunk length is set to 1.5s. ... PLDA is conducted to reduce the dimension ..."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Existing toolkits/scripts**  
"Our baseline system is publicly available at [https://github.com/MagicHub-io/MagicData-RAMC."[1](https://github.com/MagicHub-io/MagicData-RAMC.%22%5B1)]

---

**8. Challenges & Limitations**

a. **Known challenges**  
"Phenomena common in spontaneous communications, such as colloquial expressions, partial words, repetitions, and other speech disfluencies, are recorded and fully transcribed."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Limitations before training**  
"The environments are relatively quiet ... ambient noise level lower than 40 dB ... accent region is roughly balanced ... The dataset is collected indoors in small rooms."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

**9. Research Applications**

a. **Research tasks suited for**  
"Detecting the speech activities of each person in a conversation is vital to downstream tasks, like natural language processing, machine translation, etc. ... speaker diarization (SD)"[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Notable papers using dataset**  
Referenced: " Z. Yang, Y. Chen, L. Luo, R. Yang, L. Ye, G. Cheng, J. Xu, Y. Jin, Q. Zhang, P. Zhang et al., “Open source magicdata-ramc: A rich annotated mandarin conversational (ramc) speech dataset,” arXiv preprint arXiv:2203.16844, 2022."openslr+1

---

**10. Reproducibility**

a. **Dataset reproducibility**  
"We provide a detailed introduction of the dataset, rules, evaluation methods, and baseline systems, aiming to promote reproducible research in this field further."[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

b. **Version controls/update logs**  
No direct quote or mention found for version control or update logs, but the code repositories are on GitHub.[arxiv](https://arxiv.org/pdf/2208.08042.pdf)

---

_All information directly quoted from the paper arXiv:2208.08042 and referenced project websites._

1. [https://arxiv.org/pdf/2208.08042.pdf](https://arxiv.org/pdf/2208.08042.pdf)
2. [https://www.openslr.org/123/](https://www.openslr.org/123/)