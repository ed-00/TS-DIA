#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

"""
Simple kaldi_data module for handling Kaldi-style data files
"""

import subprocess
import tempfile
import os
import soundfile as sf
import numpy as np

def load_wav_scp(scp_file):
    """Load wav.scp file and return dictionary mapping uttid to wav_rxfilename"""
    wav_scp = {}
    if os.path.exists(scp_file):
        with open(scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    uttid, wav_rxfilename = parts
                    wav_scp[uttid] = wav_rxfilename
    return wav_scp

def load_spk2utt(spk2utt_file):
    """Load spk2utt file and return dictionary mapping speaker to list of utterances"""
    spk2utt = {}
    if os.path.exists(spk2utt_file):
        with open(spk2utt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    spk = parts[0]
                    utts = parts[1:]
                    spk2utt[spk] = utts
    return spk2utt

def load_segments_hash(segments_file):
    """Load segments file and return dictionary mapping uttid to (recid, start, end)"""
    segments = {}
    if os.path.exists(segments_file):
        with open(segments_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    uttid, recid, start, end = parts
                    segments[uttid] = (recid, float(start), float(end))
    return segments

def process_wav(wav_rxfilename, command):
    """Process wav file with given command and return temp filename"""
    # For simple cases, if it's just a file path, return as-is
    if not any(pipe in command for pipe in ['|', '<', '>', 'sox', 'wav-reverberate']):
        return wav_rxfilename
    
    # Create temporary file for processed audio
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    try:
        # Build command pipeline
        if wav_rxfilename.endswith('|'):
            # Input is already a command
            full_command = wav_rxfilename.rstrip('|') + ' | ' + command + ' > ' + temp_path
        else:
            # Input is a file
            full_command = f'cat "{wav_rxfilename}" | {command} > {temp_path}'
        
        # Execute command
        result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {full_command}\nError: {result.stderr}")
        
        return temp_path
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def load_wav(wav_rxfilename, start=None, end=None):
    """Load wav file and return audio data and sample rate"""
    # Handle temporary files from process_wav
    is_temp = wav_rxfilename.startswith('/tmp/')
    
    try:
        # Load audio file
        audio, sr = sf.read(wav_rxfilename)
        
        # Apply start/end if specified (in samples)
        if start is not None or end is not None:
            start = start if start is not None else 0
            end = end if end is not None else len(audio)
            audio = audio[start:end]
        
        return audio, sr
    
    finally:
        # Clean up temporary file
        if is_temp and os.path.exists(wav_rxfilename):
            os.unlink(wav_rxfilename)