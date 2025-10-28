# Makefile
# Copyright 2018-2019 Hitachi, Ltd. (author: Yusuke Fujita)
#
# install tools

# If you want to use prebuild kaldi, make KALDI=<path/to/kaldi_root>
KALDI :=
# Specify cuda root path installed in your environment
CUDA_PATH := /usr/local/cuda
CUDA_BIN := $(CUDA_PATH)/bin/nvcc
CUDA_AVAILABLE := $(shell if [ -x "$(CUDA_BIN)" ]; then echo 1; else echo 0; fi)
ifeq ($(CUDA_AVAILABLE),1)
CUDA_VERSION := $(shell $(CUDA_BIN) --version | tail -n1 | awk '{print substr($$5,0,length($$5)-1)}')
else
CUDA_VERSION :=
endif

all: kaldi miniconda3/envs/eend/bin env.sh

ifneq ($(strip $(KALDI)),)
kaldi:
	ln -s $(abspath $(KALDI)) kaldi
else
kaldi:
	git clone https://github.com/kaldi-asr/kaldi.git
	cd kaldi/tools; $(MAKE)
	cd kaldi/src; ./configure --shared --use-cuda=no; $(MAKE) depend; $(MAKE) all
endif

miniconda3.sh:
	wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $@

miniconda3: miniconda3.sh
	# -b: non-interactive install
	# -p: installed directory
	bash miniconda3.sh -b -p miniconda3

# virtual environment of python
miniconda3/envs/eend/bin: miniconda3
	yes | miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
	yes | miniconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
	miniconda3/bin/conda update -y conda
	miniconda3/bin/conda env create -f environment.yml
ifeq ($(CUDA_AVAILABLE),1)
	miniconda3/envs/eend/bin/pip install cupy-cuda$(subst .,,$(CUDA_VERSION))==6.2.0 chainer==6.2.0
else
	@echo "CUDA not detected; installing CPU-only dependencies for Chainer"
	miniconda3/envs/eend/bin/pip install chainer==6.2.0
endif
update:
	miniconda3/bin/conda env update -f environment.yml

env.sh: miniconda3/envs/eend/bin
	cp env.sh.in env.sh
	echo "export LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$$LD_LIBRARY_PATH" >> env.sh