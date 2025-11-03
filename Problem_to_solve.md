data preprocessing not working, producing wierd batches, 
based on the EEND, https://github.com/hitachi-speech/EEND/blob/b851eecd8d7a966487ed3e4ff934a1581a73cc9e/egs/mini_librispeech/v1/conf/eda/train.yaml

The expected resault is:  
Features tensors of size (B, T, F) 
* where batch is arbetrary for now, 
* time is a fixed chunk size, 
* and F is 345 dimentional 
* from sample rate of 8000,
* frame_size: 200 samples or 25 ms
* frame_shift: 80 samples or 10 ms
* num_frames: 500 samples or 62.5 ms 
* contex size, of 7
* subsampling of 10
* The input transformation is logmel23_mn
* The data is shuffled 
batch size could be set to 64 chunks. 


"""
We configured a BLSTM-based EEND method (BLSTM-EEND), as
described in [26]. The input features were 23-dimensional log-Mel-
filterbanks with a 25-ms frame length and 10-ms frame shift. Each
feature was concatenated with those from the previous seven frames
and subsequent seven frames. To deal with a long audio sequence in
our neural networks, we subsampled the concatenated features by a
factor of ten. Consequently, a (23 × 15)-dimensional input feature
was fed into the neural network every 100 ms.
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9003959
"""

What is currently happening (issues): 
1. The ego-centric lables are are computed in the ´Dataset´ class and should be precomputed. and stored as a seprate supervision set 
    requirements:
    [ ] librosa, melFbanks for consistance with other libraries (https://lhotse.readthedocs.io/en/latest/_modules/lhotse/features/librosa_fbank.html)
    [ ] The dataset should be cut into the proper size. and a corresponding supervision set created. 
    [ ] Base on the supervison cutset, each cut should be repeated for n + 1 number of speakers
        [ ] The repition should have a new ´id´ for the segment based on who is the target speaker  
        [ ] The speaker identification should reflect both the speaker and the lableling scheme, to map from class to token 
        [ ] The zero vector enrollement embedding should produce on (other_sgl, non_speech) 
        [ ] The enrollment should be equal between diffrent segment_len
        [ ] The enrollment must have qual amounts of start, ends, (center with buffer)
    [ ] the subsampling should be included in the segments (majority vote who gets the class)
    [ ] the dataloader must have batch size, thus the dataset must have len
        [ ] eager, and precomputed features are the solution maybe? 

# Knowladge gaps
this is a list of possible knowladge gaps that must be coverd:
[ ] subsamplings, effect, and process, influance by the random seed: 

"hitachi-speech/EEND/" 
```python
    def subsample(Y, T, subsampling=1):
    """ Frame subsampling
    """
    Y_ss = Y[::subsampling] # Y_ss \in R^(T/10 x F)
    T_ss = T[::subsampling] # T_ss \in R^(T/10 x S)
    return Y_ss, T_ss
```
    - this reduces the granuality the time domain 
    - skipping dimentions each subsampling value 
    - subsampling in "hitachi-speech/EEND/" is 10 

    Questions:
    1. how much are we loosing when using a subsampling of 10?
        - give that the:
            - sampling_rate: 8000 samples 
            - frame_shift: 80 samples
            - frame_size: 200 samples 
        - we can expect 10 frames per second (each 100 ms) (23 x 15) dimential vector is fed
        - (sample_rate / frame_shift) = 100
        - (sample_rate / frame_shift) / 10 = 10 samples 

    2. dose the the chunks have overlap. 
        - no start -> min(chunk_size, data_len)
         

[ ] Effect and signifiance of "logmel23" and "logmel23_mn" on the input vectors. 
[ ] not using the Sampler object, effect, benifet


# Ideas to investigate 

* Neural sampels aggrigation (T x D_model) (1 x d_model) conformer + pooling could work here 
