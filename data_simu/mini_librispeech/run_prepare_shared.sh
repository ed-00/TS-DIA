#!/bin/bash

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.
#
# This script prepares kaldi-style data sets shared with different experiments
#   - data/xxxx
#     callhome, sre, swb2, and swb_cellular datasets
#   - data/simu_${simu_outputs}
#     simulation mixtures generated with various options
# This script does NOT include the composition of train/valid/test sets.
# The composition will be done at stage 1 of ./run.sh

stage=0

# This script distributes simulated data under these directories
simu_actual_dirs=(
/workspace/TS-DIA/outputs/kaldi_mini_librispeech/simu2
)

# simulation options - defaults
simu_opts_overlap=yes
simu_opts_num_speaker=2
simu_opts_sil_scale=2
simu_opts_rvb_prob=0.5
simu_opts_num_train=100
simu_opts_min_utts=10
simu_opts_max_utts=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --overlap)
            simu_opts_overlap="$2"
            shift 2
            ;;
        --num-speaker)
            simu_opts_num_speaker="$2"
            shift 2
            ;;
        --sil-scale)
            simu_opts_sil_scale="$2"
            shift 2
            ;;
        --rvb-prob)
            simu_opts_rvb_prob="$2"
            shift 2
            ;;
        --num-train)
            simu_opts_num_train="$2"
            shift 2
            ;;
        --min-utts)
            simu_opts_min_utts="$2"
            shift 2
            ;;
        --max-utts)
            simu_opts_max_utts="$2"
            shift 2
            ;;
        --stage)
            stage="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --overlap VALUE       Enable/disable overlap (yes/no, default: yes)"
            echo "  --num-speaker VALUE   Number of speakers (default: 2)"
            echo "  --sil-scale VALUE     Silence scale factor (default: 2)"
            echo "  --rvb-prob VALUE      Reverberation probability (default: 0.5)"
            echo "  --num-train VALUE     Number of training samples (default: 100)"
            echo "  --min-utts VALUE      Minimum utterances (default: 10)"
            echo "  --max-utts VALUE      Maximum utterances (default: 20)"
            echo "  --stage VALUE         Starting stage (default: 0)"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

. path.sh
. cmd.sh
# . parse_options.sh || exit

# Display current configuration
echo "=== Simulation Configuration ==="
echo "Overlap: $simu_opts_overlap"
echo "Number of speakers: $simu_opts_num_speaker"
echo "Silence scale: $simu_opts_sil_scale"
echo "Reverberation probability: $simu_opts_rvb_prob"
echo "Number of training samples: $simu_opts_num_train"
echo "Minimum utterances: $simu_opts_min_utts"
echo "Maximum utterances: $simu_opts_max_utts"
echo "Starting stage: $stage"
echo "================================"

if [ $stage -le 0 ]; then
    echo "prepare kaldi-style datasets"
    mini_librispeech_url=http://www.openslr.org/resources/31
    mkdir -p data/local
    local/download_and_untar.sh data/local $mini_librispeech_url  dev-clean-2
    local/download_and_untar.sh data/local $mini_librispeech_url train-clean-5
    if [ ! -f data/dev_clean_2/.done ]; then
        local/data_prep.sh data/local/LibriSpeech/dev-clean-2 data/dev_clean_2 || exit
        touch data/dev_clean_2/.done
    fi
    if [ ! -f data/train_clean_5/.done ]; then    
        local/data_prep.sh data/local/LibriSpeech/train-clean-5 data/train_clean_5
        touch data/train_clean_5/.done
    fi
    if [ ! -d data/musan_bgnoise ]; then
        tar xzf musan_bgnoise.tar.gz
    fi
    if [ ! -f data/simu_rirs_8k/.done ]; then
        mkdir -p data/simu_rirs_8k
        if [ ! -e sim_rir_8k.zip ]; then
            wget --no-check-certificate http://www.openslr.org/resources/26/sim_rir_8k.zip
        fi
        unzip sim_rir_8k.zip -d data/sim_rir_8k
        find $PWD/data/sim_rir_8k -iname "*.wav" \
            | awk '{n=split($1,A,/[\/\.]/); print A[n-3]"_"A[n-1], $1}' \
            | sort > data/simu_rirs_8k/wav.scp
        awk '{print $1, $1}' data/simu_rirs_8k/wav.scp > data/simu_rirs_8k/utt2spk
        utils/fix_data_dir.sh data/simu_rirs_8k
        touch data/simu_rirs_8k/.done
    fi
fi

simudir=data/simu
if [ $stage -le 1 ]; then
    echo "simulation of mixture"
    mkdir -p $simudir/.work
    random_mixture_cmd=random_mixture_nooverlap.py
    make_mixture_cmd=make_mixture_nooverlap.py
    if [ "$simu_opts_overlap" == "yes" ]; then
        random_mixture_cmd=random_mixture.py
        make_mixture_cmd=make_mixture.py
    fi

    for simu_opts_sil_scale in 2; do
        for dset in train_clean_5 dev_clean_2; do
            n_mixtures=500
            simuid=${dset}_ns${simu_opts_num_speaker}_beta${simu_opts_sil_scale}_${n_mixtures}
            # check if you have the simulation
            if ! validate_data_dir.sh --no-text --no-feats $simudir/data/$simuid; then
                # random mixture generation
                $simu_cmd $simudir/.work/random_mixture_$simuid.log \
                    $random_mixture_cmd --n_speakers $simu_opts_num_speaker --n_mixtures $n_mixtures \
                    --speech_rvb_probability $simu_opts_rvb_prob \
                    --sil_scale $simu_opts_sil_scale \
                    data/$dset data/musan_bgnoise data/simu_rirs_8k \
                    \> $simudir/.work/mixture_$simuid.scp
                nj=1
                mkdir -p $simudir/wav/$simuid
                # distribute simulated data to $simu_actual_dir
                split_scps=
                for n in $(seq $nj); do
                    split_scps="$split_scps $simudir/.work/mixture_$simuid.$n.scp"
                    mkdir -p $simudir/.work/data_$simuid.$n
                    actual=${simu_actual_dirs[($n-1)%${#simu_actual_dirs[@]}]}/$simudir/wav/$simuid/$n
                    mkdir -p $actual
                    ln -nfs $actual $simudir/wav/$simuid/$n
                done
                utils/split_scp.pl $simudir/.work/mixture_$simuid.scp $split_scps || exit 1

                $simu_cmd --max-jobs-run 32 JOB=1:$nj $simudir/.work/make_mixture_$simuid.JOB.log \
                    $make_mixture_cmd --rate=8000 \
                    $simudir/.work/mixture_$simuid.JOB.scp \
                    $simudir/.work/data_$simuid.JOB $simudir/wav/$simuid/JOB
                utils/combine_data.sh $simudir/data/$simuid $simudir/.work/data_$simuid.*
                steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
                    $simudir/data/$simuid/utt2spk $simudir/data/$simuid/segments \
                    $simudir/data/$simuid/rttm
                utils/data/get_reco2dur.sh $simudir/data/$simuid
            fi
        done
    done
fi