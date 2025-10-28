# Essential Reading List with Links


## 1. Dataset List

1. **AMI Corpus**  [[AMI Dataset (information card)]](../datasets/AMI-dataset-information-card.md)
   Carletta, J., McCowan, I., Bourban, S., et al. “The AMI Meeting Corpus: A Pre-announcement.” Machine Learning for Multimodal Interaction, 2005.  
   https://groups.inf.ed.ac.uk/ami 

2. **ICSI Meeting Corpus**  [[ICSI dataset overview]](../datasets/ICSI-dataset-overview.md)
   Janin, A., Baron, D., Edwards, J., et al. “The ICSI Meeting Corpus.” *Proceedings of ICASSP*, 2003.  
   https://catalog.ldc.upenn.edu/LDC2004S02  

3. **AISHELL-4**  [[AISHELL-4 dataset overview]](../datasets/AISHELL-4-dataset-overview.md)
   Zhang, X., Shi, L., Wu, J., et al. “AISHELL-4: An Open-Source Dataset for Speech Separation and Recognition in Conference Scenario.” *Interspeech 2021*.  
   https://arxiv.org/abs/2104.03603  

4. **MSDWild**  [[MSDWild]](../datasets/MSDWild.md)
   Liu, Y., Li, Z., Zhang, H., et al. “MSDWILD: Multi-Modal Speaker Diarization Dataset in the Wild.” *Interspeech 2022*.  
   https://www.isca-archive.org/interspeech_2022/liu22t_interspeech.pdf  

5. **VoxConverse**  [[VoxConverse dataset overview]](../datasets/VoxConverse-dataset-overview.md)
   Chung, J. S., Nixon, A. L., Remelli, E., et al. “Spot the Conversation: Speaker Diarisation in the Wild.” *Interspeech 2020*.  
	https://arxiv.org/pdf/2007.01216

6. **AVA-AVD** [[AVA-AVD]](../datasets/AVA-AVD.md) 
   Xu, Z., Song, Y., Huang, J., et al. “AVA-AVD: Audio-Visual Speaker Diarization in the Wild.” *ACM Multimedia 2021*.  
   https://arxiv.org/abs/2111.14448  

7. **Earnings-21**  [[Earnings-21]](../datasets/Earnings-21.md)
   Del Rio, E., Sainath, T. N., Shatilov, D., et al. “Earnings-21: A Practical Benchmark for ASR and Diarization in Financial Earnings Calls.” *Interspeech 2021*.  
   https://arxiv.org/abs/2104.11348  

8. **LibriheavyMix**  [[LibriheavyMix]](../datasets/LibriheavyMix.md)
   Jin, X., Zhang, Y., Wei, S., et al. “LibriheavyMix: A 20,000-Hour Dataset for Single-Channel Overlapped Speech Separation and Diarization.” *Interspeech 2024*.  
   https://arxiv.org/abs/2409.00819  

9. **M3SD**  [[M3SD Dataset]](../datasets/M3SD-Dataset.md) 
	Wu, S. (2025). M3SD: Multi-modal, Multi-scenario and Multi-language Speaker Diarization Dataset. _arXiv preprint arXiv:2506.14427v2_.

10. **Ego4D Audio Diarization**  [[Ego4D Audio Diarization]](../datasets/Ego4D-Audio-Diarization.md)
    Lin, T. H., et al. “Ego4D: Audio-only Speaker Diarization in First-Person Video.” *CVPR Workshops 2023*.  
    https://ego4d-data.org/  



## 2. Dataset overview

A quick comparison of major diarization datasets:

| **Dataset**         | **Type**   | **Hours** | **Speakers**    | **Rate**  |
|---------------------|------------|-----------|-----------------|-----------|
| AMI                 | Meeting    | 100       | Multi           | 16kHz     |
| ICSI                | Meeting    | 75        | Multi           | 16kHz     |
| AISHELL-4           | Meeting    | 120       | Multi           | 16kHz     |
| MSDWild             | AV         | 70        | Mixed           | 16kHz     |
| Earnings-21         | Calls      | 40        | Multi           | Mixed     |
| VoxConverse         | AV         | 70        | Multi           | Mixed     |
| AVA-AVD             | AV         | 30        | Multi           | 16kHz     |
| M3SD (pseudo-labels)| AV         | 770       | Multi           | Mixed     |
| Ego4D               | AV         | 3900      | Multi           | Mixed     |
| LibriheavyMix       | Synth      | 20000     | Simulated       | 16kHz     |

- **Type**: AV = Audio-Visual, Synth = Synthetic mixtures, Calls = Telephone, Meeting = Conference/meeting room.
- **Speakers**: "Multi" = multiple real speakers per session; "Simulated" = synthetic mixtures; "Mixed" = various.
- **Rate**: Audio sampling rate (may vary within some datasets).

For a detailed review template, see [`docs/datasets/review-template.md`](../datasets/review-template.md).


## 3. Data standardization

The audio, video, and annotations should be standardized to make it easier to compare and use across datasets. Furthermore the annoation must be restrucured from a global diarization format to a local diarization format to suit the [problem formulation](problem-formulation.md). 


### Audio standardization

1. The audio should be standardized to 8kHz sampling rate. 

2. All audio formats should be converted to wav format more specifically mono wav format.

3. Normalization should be done to the audio to ensure that the audio is not too loud or too quiet.

### Annotation standardization

to produce the local diarization format, the annotations must be restructured from a global diarization format to a local diarization format. 

For each audio file with $n$ number of speakers $S = \{s_1, \dots, s_n\}$, the annoation will be:

$$
Y^{s} = [y_1, y_2, \dots, y_T] 
$$
where $y_{t} \in C$

and 
$$
\mathcal{C} = \{\texttt{ts},\ \texttt{ts\_{ovl}},\ \texttt{others\_{sgl}},\ \texttt{others\_{ovl}},\ \texttt{ns}\}
$$

Thus producing from a single input $X$ and traget $Y$ lables set $n$ number of subjectiv lables for each speaker. 


### Honorable mentions (not in the table below)

- **AliMeeting**  [[AliMeeting dataset overview]](../datasets/AliMeeting-dataset-overview.md)  
  Wang, W., Chen, X., Zhang, G., et al. “AliMeeting: A Chinese Multi-Channel Meeting Corpus for Speech Processing.” *ICASSP 2022*.  
  https://arxiv.org/abs/2104.03603  

- **Synthetic LibriSpeech Corpus**  [[Synthetic LibriSpeech Corpus]](../datasets/Synthetic-LibriSpeech-Corpus.md)  
  Suendermann, D., et al. “A Free Synthetic Corpus for Speaker Diarization Research.” *SPECOM 2018*.  
  http://suendermann.com/su/pdf/specom2018d.pdf  

- **DIHARD III**  [[DIHARD III]](../datasets/DIHARD-III.md)  
  Farnoosh, A., Sell, G., Mandl, T., et al. “The Third DIHARD Diarization Challenge.” *arXiv:2012.01477*, 2020.  
  https://dihardchallenge.github.io/dihard3/  

- **DISPLACE 2024**  [[DISPLACE 2024]](../datasets/DISPLACE-2024.md)  
  Durmuş, E., et al. “The Second DISPLACE Challenge: DIarization in Conversational Environments.” *arXiv:2406.09494*, 2024.  
  https://displace2024.github.io  

- **CHiME-6**  [[CHiME-6 Challenge]](../datasets/CHiME-6-Challenge.md)  
  Watanabe, S., et al. “The 6th CHiME Challenge: Multispeaker Speech in Home Environments.” *arXiv:2004.09249*, 2020.  
  https://chimechallenge.github.io/chime6/  

- **CALLHOME**  
  Imseng, D., et al. “Automatic Speaker Segmentation for Conversational Telephone Speech.” *Interspeech 2007*.  
  https://huggingface.co/datasets/diarizers-community/callhome  

- **Fisher English**  
  “Fisher Corpus: LDC2004T19.”  
  https://catalog.ldc.upenn.edu/LDC2004T19  

- **Switchboard**  
  “Switchboard-1 Release 2.”  
  https://catalog.ldc.upenn.edu/LDC97S62  

- **CSSD**  [[CSSD]](../datasets/CSSD.md)  
  Han, B., et al. “The Conversational Short-phrase Speaker Diarization (CSSD) Challenge.” *arXiv:2208.08042*, 2022.  
  https://arxiv.org/abs/2208.08042  

- **SDBench**  
  Durmuş, E., et al. “SDBench: A Comprehensive Benchmark Suite for Speaker Diarization.” *arXiv:2507.16136*, 2025.  
  https://github.com/argmaxinc/SDBench  

***