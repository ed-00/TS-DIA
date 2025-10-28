# Task-independent environmental variables
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -z "${KALDI_ROOT:-}" ]; then
	if [ -d "${SCRIPT_DIR}/kaldi" ]; then
		KALDI_ROOT="$( cd "${SCRIPT_DIR}/kaldi" && pwd -P )"
	else
		echo >&2 "KALDI_ROOT is unset and ${SCRIPT_DIR}/kaldi does not exist -> Exit!"
		exit 1
	fi
else
	KALDI_ROOT="$( cd "${KALDI_ROOT}" && pwd -P )"
fi
export KALDI_ROOT

[ -f "${KALDI_ROOT}/tools/env.sh" ] && . "${KALDI_ROOT}/tools/env.sh"

export PATH="$PWD/utils:${KALDI_ROOT}/tools/openfst/bin:${KALDI_ROOT}/tools/sph2pipe_v2.5:${KALDI_ROOT}/tools/sctk/bin:$PWD:$PATH"

COMMON_PATH="${KALDI_ROOT}/tools/config/common_path.sh"
if [ ! -f "${COMMON_PATH}" ]; then
	echo >&2 "The standard file ${COMMON_PATH} is not present -> Exit!"
	exit 1
fi
. "${COMMON_PATH}"

MINICONDA_BIN="${SCRIPT_DIR}/miniconda3/envs/eend/bin"
if [ -d "${MINICONDA_BIN}" ]; then
	export PATH="${MINICONDA_BIN}:$PATH"
fi

EEND_BIN="$( cd "${SCRIPT_DIR}/.." && pwd )/eend/bin"
if [ -d "${EEND_BIN}" ]; then
	export PATH="${EEND_BIN}:$PATH"
fi

UTILS_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )/utils"
if [ -d "${UTILS_DIR}" ]; then
	export PATH="${UTILS_DIR}:$PATH"
fi

PROJECT_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"export LD_LIBRARY_PATH=/usr/local/cuda/lib64:
