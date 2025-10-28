
# Data prep plan 
Week 43, the transformer model is implementd, the data need to be preped to start training on the diarization data. 

[x]  Decide on a preprocessing standard for all the datasets 
    [x] Choose sample rate (8khz becasue many other studies use that) 
    [x] Choose window and overlap, 25ms window 10ms overlap (AED-EEND uses that) 
    [x] Choose feature dimentions, 345 feature dimentitions ie 21 feature bins
    [x] Choose locaiton for the data (/sbt-fast/dia-df/TS-DIA-2025-10/TS-DIA/data) 

[ ] Create a diarization labels converstion function 
    [ ] For each speaker in a sample, repeat the fueatures N number of speaker time
    [ ] Convert diarization lables from R^(SxT) to (C x T)
    [ ] Store the altered lables while keeping the audio files untouched 

[x] Run the prep function or all the datasets 
    <!-- [ ] LibriheavyMix 
        [x] yaml
        [ ] download
        [ ] validate -->
    [x] Ami 
        [x] yaml
        [x] download
        [x] validate
    [x] ICSI 
        [x] yaml
        [x] download
        [x] validate
    [x] AISHELL-4
        [x] yaml
        [x] download
        [x] validate
    [x] VoxConverse
        [x] yaml
        [x] download
        [x] validate
    [x] Ava-Avd
        [x] yaml
        [x] download
        [x] validate 
    [x] mswild 
        [x] yaml
        [x] download
        [x] validate 

    [ ] DIHARD III
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] CHiME-6
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] LibriCSS
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] TED-LIUM v2/v3
        [ ] yaml
        [ ] download
        [ ] validate 
    [ ] VoxCeleb1 & 2
        [ ] yaml
        [ ] download
        [ ] validate 
    <!-- [ ] Ego-4d
        [x] Apply for access again
        [x] yaml
        [ ] download
        [ ] validate -->


[ ] Validate and Create descriptions for all the datasets. 

[ ] Write the Datasets sections in the paper. 

[ ] prep data simiulation useing mini_librispeech
    [ ] Setup kaldi continer 
    [ ] clone mini_librispeech recipe
    [ ] prepare datasets with varing overlap threshold 
        [ ] train 20k h 
        [ ] 1 - 4 speakers N(1, 4)
            [ ] overlap threshold modification 
            [ ] 100 h val for each combo

    [ ] ensure lhosete compatability 

[ ] setup training experiments 
    [ ] softmax attention
    [ ] linear attention 

 

