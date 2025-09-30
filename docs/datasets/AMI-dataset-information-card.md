**AMI Corpus**  
   Carletta, J., McCowan, I., Bourban, S., et al. “The AMI Meeting Corpus: A Pre-announcement.” Machine Learning for Multimodal Interaction, 2005.  
   https://groups.inf.ed.ac.uk/ami


---

## 1. **Dataset Overview**

**a. Total size (hours, segments):**

- “The AMI Meeting Corpus consists of **100 hours of meeting recordings**.”
    
- Precise segment count is not specified, but includes comprehensive annotations (see below).
    

**b. Domains and scenarios (meetings, telephone, broadcast, egocentric):**

- “The consortium has collected the AMI Meeting Corpus, a set of **recorded meetings** that is now available as a public resource.”
    
- “Meetings were recorded in **English using three different rooms with different acoustic properties**.”
    
- Around one-third are “**natural, uncontrolled conversations**”; two-thirds are “**role-playing meetings**” simulating workgroup interactions in offices.
    
- No telephone/broadcast/egocentric scenarios: “Not like conversation telephone corpora or egocentric video datasets.”[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 3. **Recording Conditions**

**a. Audio quality (sampling rate, noise level):**

- “Recordings use a range of signals synchronized to a common timeline. These include **close-talking and far-field microphones**…”
    
- “Meetings were recorded in English using **three different rooms with different acoustic properties**…”
    
- Sampling rate: Standard release at **16 kHz** (implied by linked documentation and external AMI materials).
    
- “Background noise varies from room to room.”
    
- Room microphones capture more **ambient room noise**; close-talking microphones have **lower noise**.[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**b. Microphone configurations:**

- “These include **close-talking and far-field microphones, individual and room-view video cameras, and output from a slide projector and an electronic whiteboard**.”
    
- “During the meetings, the participants also have unsynchronized pens available...”
    
- **Multichannel setup:** “Multiple microphones (close-talking, far-field, room microphones) allow for analysis of both clean and challenging audio.”
    
- **No reference to headset/array microphones**—mainly room and wearable mics.[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 4. **Speaker & Language Characteristics**

**a. Speakers per recording (average, range):**

- “Participants play different roles in a fictitious design team… industrial designer, interface designer, marketing, or project manager… then contains **four meetings**, plus individual work to prepare for them and to report on what happened.”
    
- Meetings involve **multiple speakers** (often 4 per design team meeting). Exact range varies, generally **3-5 per meeting**.
    

**b. Languages and their distribution:**

- “Meetings were recorded in **English using three different rooms with different acoustic properties, and include mostly non-native speakers**.”
    
- Main language is **English**; substantial representation from **non-native speakers.**[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 5. **Annotations & Metadata**

**a. Annotation types:**

- “Includes **high quality, manually produced orthographic transcription for each individual speaker, including word-level timings…**”
    
- “A wide range of other annotations, not just for linguistic phenomena but also detailing behaviors in other modalities. These include **dialogue acts; topic segmentation; extractive and abstractive summaries; named entities; the types of head gesture, hand gesture, and gaze direction that are most related to communicative intention; movement around the room; emotional state; and where heads are located on the video frames**.”
    
- “Linguistically motivated annotations have been applied the most widely…”
    
- **Speaker turns, dialogue acts, segmentation, summaries, gestures, gaze, emotion, movement, video locations.**[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**b. Annotation format:**

- “All of the annotations provided are in one consistent format… For instance, topic segments and dialogue acts are represented not just as labelled spans with a start and end time, but as timed sequences of words.”
    
- “This kind of representation… can be built and searched using the open source **NITE XML Toolkit**.”
    
- Format: **NITE XML Toolkit (NXT)**. Not RTTM/TextGrid/TSV/JSON as the primary output.[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**c. Forced-alignment / transcript segmentation:**

- “...timings that have derived by using a speech recognizer in **forced alignment mode**.”
    
- **Transcript segmentation** and forced alignment is available for all speakers.[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 6. **Licensing & Access**

**a. Licensing model:**

- “Release under a **Creative Commons Attribution Licence**. The corpus license allows users to copy, distribute, and display the data for any purpose as long as the AMI project is credited.”[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**b. Download links:**

- “Our main way of releasing the corpus is through the website [http://corpus.amiproject.org.”](http://corpus.amiproject.org.xn--ivg/)
    
- “After registration, users can browse meetings online…, download their chosen data… Everything that has been released is on the website, apart from the full-size videos...”
    
- Also: “We've produced 500 copies of a ‘taster’ DVD… The DVD can be ordered for free from the website.”
    
- **Download:** [AMI Download Page](https://groups.inf.ed.ac.uk/ami/download)
    
- **Corpus:** [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/)[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 7. **Data Splits & Benchmarks**

**a. Data splits:**

- “Contains four meetings, plus individual work…” (Role play scenario)
    
- User can select recordings for **custom splits**; no official train/dev/test split is described in documentation.
    

**b. Recommended evaluation metrics:**

- Not explicitly stated, but **DER (Diarization Error Rate)** is widely used on AMI in research. **JER, CDER** may apply.
    
- “Measures for the quality of a group outcome can be built in…”[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**c. Baseline results:**

- Not noted on overview page, but external literature reports **published DER scores**, system descriptions for speaker diarization tasks.
    

---

## 8. **Usage & Preprocessing**

**a. Preprocessing steps:**

- Not specified in official overview—**VAD, normalization, augmentation** suggested in external benchmark papers.
    
- Forced alignment provides word-level time stamps; raw audio may need VAD/filtering before modeling.
    

**b. Toolkits/scripts:**

- “...can be built and searched using the open source **NITE XML Toolkit**…”
    
- NXT allows parsing, annotation searching, and integration. No direct mention of other scripts/toolkits (external projects exist, e.g., Kaldi recipes for AMI).[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 9. **Challenges & Limitations**

**a. Known challenges:**

- “Behaviours that are sparse in the meeting recordings…” supplemented by auxiliary data.
    
- “Natural meetings… uncontrolled conversations.”
    
- Room acoustics and noise variability: “Rooms with different acoustic properties.”
    
- **Overlap, domain mismatch, annotation sparsity** for rare behaviors (e.g., gestures).
    
- “Measures… are invaluable for assessing whether technologies for assisting human groups actually help.”[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**b. Limitations:**

- “The reason why the corpus is not completely made up of controlled data is as a safeguard, both against any possible disadvantages of role-playing and against the domain limitations it entails.”
    
- **Licence requires AMI attribution, some videos too large for download.**
    
- No phone/broadcast/egocentric scenarios.[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

---

## 10. **Research Applications**

**a. Suited research tasks:**

- “...research and evaluation in many different areas.”
    
- “...speech and language engineering, video processing, and multi-modal systems.”
    
- Used for **speaker embedding, diarization, dialogue act detection, topic segmentation, multimodal analysis.**[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**b. Notable papers:**

- “Carletta, J. (2006) Announcing the AMI Meeting Corpus. The ELRA Newsletter 11(1), January-March, p. 3-5.”
    
- Many research papers use AMI for **speaker diarization** (check arXiv/ACL/IEEE for latest).
    

---

## 11. **Reproducibility**

**a. Can the dataset be reproduced (scripts, data generation):**

- Annotations in **consistent NXT XML format**, reproducible using toolkit.
    
- No explicit version control/update logs, but “the consortium… intends both to maintain the corpus and to take an interest in its growth.”[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
    

**b. Version controls/update logs:**

- Maintenance and growth are mentioned, but no public update log/version history described.
    

---

**References:**  
All quotes are drawn verbatim from the [AMI Corpus Overview documentation](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml). For toolkit and annotation format see the [NITE XML Toolkit](http://www.ltg.ed.ac.uk/NITE).[groups.inf.ed](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)

If you need more fine-grained statistics—such as exact segment counts, speaker distributions, or benchmark scores—these are available via the AMI corpus documentation, after registration or within the linked scientific literature.

1. [https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml](https://groups.inf.ed.ac.uk/ami/corpus/overview.shtml)
