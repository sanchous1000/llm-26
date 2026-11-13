#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=2048
export OLLAMA_KEEP_ALIVE=1m
export OLLAMA_NO_CLOUD=1

exec ollama serve
