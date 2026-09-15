# shellcheck shell=bash
# Earmark development environment. Source it, do not execute it:
#
#     source scripts/env.sh
#
# Works from bash and zsh. Points every large download cache (Hugging Face, torch hub,
# uv, pip, npm, Playwright browsers) into ./.cache so it stays inside the repository,
# is ignored by git and can be wiped with `make clean-caches`. Tokens are never set here:
# HF_TOKEN comes from your shell, Kaggle Secrets or Colab Secrets.

if [ -n "${BASH_VERSION:-}" ]; then
  _earmark_env_src="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  _earmark_env_src="${(%):-%x}"
else
  _earmark_env_src="$0"
fi

EARMARK_ROOT="$(cd "$(dirname "$_earmark_env_src")/.." && pwd -P)"
unset _earmark_env_src
export EARMARK_ROOT

export EARMARK_CACHE="$EARMARK_ROOT/.cache"
export HF_HOME="$EARMARK_CACHE/huggingface"
export TORCH_HOME="$EARMARK_CACHE/torch"
export UV_CACHE_DIR="$EARMARK_CACHE/uv"
export PIP_CACHE_DIR="$EARMARK_CACHE/pip"
export npm_config_cache="$EARMARK_CACHE/npm"
export PLAYWRIGHT_BROWSERS_PATH="$EARMARK_CACHE/ms-playwright"

mkdir -p "$HF_HOME" "$TORCH_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$npm_config_cache" \
  "$PLAYWRIGHT_BROWSERS_PATH"

# Make `python -m earmark...` work without installing the package.
case ":${PYTHONPATH:-}:" in
  *":$EARMARK_ROOT/python:"*) ;;
  *) export PYTHONPATH="$EARMARK_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
