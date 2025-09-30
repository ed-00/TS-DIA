
**VoxConverse**  
   Chung, J. S., Nixon, A. L., Remelli, E., et al. “Spot the Conversation: Speaker Diarisation in the Wild.” *Interspeech 2020*. 
---

https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip
https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip
## 1. **Dataset Overview**

**a. Total size (hours, segments):**

- “VoxConverse currently contains 70 hours of annotated video, but we are in the process of scaling up.”
    
- “The development set...216 multispeaker videos covering 1,218 minutes with 8,268 speaker turns annotated. The test set contains...232 videos covering 2,612 minutes.”
    
- “Videos vary in length from 22 seconds to 20 minutes.”
    
- “On average, 91% of the video time contains speech, and 3–4% of this contains speech where one speaker overlaps with another speaker.”
    

**b. Domains and scenarios:**

- “Videos included in the dataset are shot in a large number of challenging multi-speaker acoustic environments, including political debates, panel discussions, celebrity interviews, comedy news segments and talk shows.”
    
- “We propose an automatic audio-visual diarisation method for YouTube videos.”
    
- “collected from ‘in the wild’ videos...”
    

---

## 2. **Recording Conditions**

**a. Audio quality (sampling rate, noise level):**

- “This provides a number of background degradations, including dynamic environmental noise with some speech-like characteristics, such as laughter and applause.”
    
- “challenging background conditions”
    
- (Sampling rate is not stated explicitly.)
    

**b. Mic configurations (single, multichannel, headset, array):**

- “Our dataset is audio-visual, and contains face detections and tracks as part of the annotation.”
    
- (No explicit reference to microphone configuration, but web video sources imply variable setups.)
    

---

## 3. **Speaker & Language Characteristics**

**a. Speakers per recording (average, range):**

- “Unlike other domains such as telephony, each video has on average between 4 and 6 speakers, with one video in the dataset having 21 speakers.”
    

**b. Languages represented and distribution:**

- (Languages in the dataset are not specified in the paper.)
    

---

## 4. **Annotations & Metadata**

**a. Annotation types (labels, errors):**

- “Our dataset...contains face detections and tracks as part of the annotation.”
    
- “overlapping speech...included in evaluation.”
    
- “Diarisation labels”
    
- “speech segments are split when pauses are greater than 0.25 seconds.”
    
- “known speakers are named...marked boundaries are within 0.1 seconds of the true boundary.”
    
- “A subset...labelled independently by two annotators...Diarisation error rate between...is approximately 1%...”
    

**b. Annotation format (RTTM, TextGrid, etc.):**

- “annotated using a customised version of the VGG Image Annotator (VIA)”
    
- (Specific file format is not mentioned.)
    

**c. Forced-alignment or transcript segmentation:**

- “Anything that can be transcribed, including short utterances such as ‘yes’ and ‘right’, are considered to be speech.”
    
- (Forced-alignment is not referenced.)
    

---

## 5. **Licensing & Access**

**a. Licensing model:**

- “...will be released publicly to the research community free of charge.”
    
- (No mention of specific license or data use agreement.)
    

**b. Download links for audio, annotations, docs:**

- “[http://www.robots.ox.ac.uk/˜vgg/data/voxconverse”](http://www.robots.ox.ac.uk/%CB%9Cvgg/data/voxconverse%E2%80%9D)
    

---

## 6. **Data Splits & Benchmarks**

**a. Training/dev/test splits:**

- “The development set...216 multispeaker videos...The test set contains...232 videos...”
    

**b. Evaluation metrics:**

- “We use the diarisation error rate (DER), defined as the sum of missed speech (MS), false alarm speech (FA), and speaker misclassification error (speaker confusion, SC).”
    

**c. Baseline results published:**

- “Table 3 shows the results of all the evaluations. Our audio-visual method obtains a DER much lower than the audio-only state-of-the-art baselines...”
    
- Sample DER: “DIHARD 2019 baseline...DER 23.8%, DIHARD...w/ SE...DER 20.2%, Ours (SyncNet ASD only)...DER 10.4%, Ours (AVSE ASD only)...DER 12.4%, Ours (proposed)...DER 7.7”
    

---

## 7. **Usage & Preprocessing**

**a. Preprocessing steps:**

- “Speech segments are obtained using VAD, and divided into short overlapping segments (1.5s with 0.75s overlap).”
    
- “Speaker embeddings are extracted using the x-vector system...”
    
- “Segments are then grouped using agglomerative hierarchical clustering (AHC)...”
    

**b. Existing toolkits/scripts:**

- “We use this public code^1 as an audio-only baseline.”
    
- “(1) [https://github.com/iiscleap/DIHARD](https://github.com/iiscleap/DIHARD) 2019 baseline alltracks”
    
- “(2) [https://github.com/staplesinLA/denoising](https://github.com/staplesinLA/denoising) DIHARD18”
    

---

## 8. **Challenges & Limitations**

**a. Known challenges:**

- “lack of a fixed domain...large number of speakers...rapid exchanges with cross-talk, and background degradation consisting of channel noise, laughter and applause.”
    
- “The most common is non-visible speech segment assigned to the wrong speaker, but false alarm of the VAD and missed overlapped speech are also relatively common.”
    

**b. Limitations before training:**

- “It is almost impossible to manually annotate the segments in our dataset without the video.”
    
- “Even with the video, it can take 10 times the video duration to annotate segments to satisfactory quality if starting from scratch...”
    
- “license restrictions” not referenced for access.
    

---

## 9. **Research Applications**

**a. Research tasks suited for:**

- “Speaker diarisation is the challenging task of breaking up multispeaker video into homogeneous single speaker segments...”
    
- “audio-visual diarisation techniques...active speaker detection...speaker verification...”
    

**b. Notable papers using this dataset:**

- (Not specified, but future challenge planned: “The data will be used in the second VoxCeleb Speaker Recognition Challenge in October 2020.”)
    

---

## 10. **Reproducibility**

**a. Can it be reproduced (scripts, generation):**

- “Our pipeline is scalable...a semi-automatic dataset creation pipeline which consists of human annotation and automatic diarisation.”
    
- (No open scripts directly referenced for full dataset creation.)
    

**b. Version controls or update logs:**

- (Not stated.)
    

---

**References to direct statements from the paper above are denoted by paragraph.**

1. [https://arxiv.org/pdf/2007.01216.pdf](https://arxiv.org/pdf/2007.01216.pdf)