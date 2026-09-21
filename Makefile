# Earmark developer entry points. Works with GNU make 3.81 (macOS) and newer.
# Override any variable on the command line, e.g. `make test PYTHON=python3`.

PYTHON ?= .venv/bin/python
CMAKE ?= $(if $(wildcard .venv/bin/cmake),.venv/bin/cmake,cmake)
CTEST ?= $(if $(wildcard .venv/bin/ctest),.venv/bin/ctest,ctest)
# CMAKE_MAKE_PROGRAM must be a path, so resolve a PATH ninja to its absolute location.
NINJA ?= $(if $(wildcard .venv/bin/ninja),$(abspath .venv/bin/ninja),$(or $(shell command -v ninja 2>/dev/null),ninja))
NODE ?= node

ENGINE_DIR ?= engine
ENGINE_BUILD_DIR ?= engine/build
ENGINE_BUILD_TYPE ?= RelWithDebInfo
ENGINE_CMAKE_ARGS ?=
# Sanitizer build (Linux CI; the macOS 26 ASan runtime hangs before main, so on a Mac
# use `make engine-sanitize SANITIZERS=-DEARMARK_UBSAN=ON`).
SANITIZE_BUILD_DIR ?= engine/build-asan
SANITIZERS ?= -DEARMARK_ASAN=ON -DEARMARK_UBSAN=ON
WASM_OUT_DIR ?= engine/build-wasm

# PyTorch-stream dev runner (python/earmark/eval/dev_runner.py).
DEV_RUNNER ?= earmark.eval.dev_runner
EVAL_ARGS ?=
# Groups of scripts/fetch_eval_data.sh that the gates need.
GATE_ASSETS ?= vbd gtcrn

PYTEST_ARGS ?=
CACHE_DIRS = .cache/huggingface .cache/torch .cache/uv .cache/pip .cache/npm .cache/ms-playwright

.PHONY: help test gates fetch-eval-data codegen contract-check goldens goldens-check \
	engine engine-sanitize wasm eval-dev demo disk-guard clean-caches

help:
	@echo "make test              CPU unit tests (excludes the slow and gate markers)"
	@echo "make gates             Suite B reproduction gates (fetches VB-DEMAND + GTCRN first)"
	@echo "make fetch-eval-data   download and sha256-verify the evaluation assets"
	@echo "make codegen           regenerate constants from contract/signal.yaml"
	@echo "make contract-check    fail if the generated constants are stale"
	@echo "make goldens           regenerate the engine golden tensors"
	@echo "make goldens-check     fail if the committed goldens are stale"
	@echo "make engine            configure, build and ctest the C++ engine (Ninja)"
	@echo "make engine-sanitize   the same under ASan + UBSan (Linux; see SANITIZERS)"
	@echo "make wasm              build earmark.wasm with em++ and smoke-test it in node (CI)"
	@echo "make eval-dev EVAL_ARGS=\"--config M --checkpoint CKPT --manifest DEV/manifest.parquet ...\""
	@echo "                       score Earmark-Synth dev with the PyTorch-stream runner"
	@echo "make demo MODEL=weights.emwb  serve the browser demo on 127.0.0.1 (see web/README.md)"
	@echo "make disk-guard        fail when less than 5 GiB is free"
	@echo "make clean-caches      delete ./.cache downloads and Python caches"

test:
	CUDA_VISIBLE_DEVICES= $(PYTHON) -m pytest -q -m "not slow and not gate" $(PYTEST_ARGS)

# The gate tests fail (never skip) when their data is missing, so fetch it first.
# The fetch is cheap when the files are already present and verified.
gates: fetch-eval-data
	$(PYTHON) -m pytest -q -m gate $(PYTEST_ARGS)

fetch-eval-data:
	scripts/fetch_eval_data.sh $(GATE_ASSETS)

codegen:
	$(PYTHON) contract/codegen.py

contract-check:
	$(PYTHON) contract/codegen.py --check

goldens:
	PYTHONPATH=python $(PYTHON) -m earmark.export.golden

# The browser demo. MODEL is the .emwb blob (its .json sits beside it); DEMO_ARGS passes
# the rest, e.g. DEMO_ARGS="--port 9000 --wasm path/to/earmark.wasm".
demo:
	@test -n "$(MODEL)" || { echo "set MODEL=<weights.emwb> (see web/README.md)"; exit 1; }
	PYTHONPATH=python $(PYTHON) -m earmark.demo --model "$(MODEL)" $(DEMO_ARGS)

goldens-check:
	PYTHONPATH=python $(PYTHON) -m earmark.export.golden --check

engine:
	$(CMAKE) -S $(ENGINE_DIR) -B $(ENGINE_BUILD_DIR) -G Ninja -DCMAKE_MAKE_PROGRAM=$(NINJA) -DCMAKE_BUILD_TYPE=$(ENGINE_BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $(ENGINE_CMAKE_ARGS)
	$(CMAKE) --build $(ENGINE_BUILD_DIR)
	$(CTEST) --test-dir $(ENGINE_BUILD_DIR) --output-on-failure

engine-sanitize:
	$(CMAKE) -S $(ENGINE_DIR) -B $(SANITIZE_BUILD_DIR) -G Ninja -DCMAKE_MAKE_PROGRAM=$(NINJA) -DCMAKE_BUILD_TYPE=Debug -DEARMARK_BUILD_SHARED=OFF $(SANITIZERS) $(ENGINE_CMAKE_ARGS)
	$(CMAKE) --build $(SANITIZE_BUILD_DIR)
	$(CTEST) --test-dir $(SANITIZE_BUILD_DIR) --output-on-failure

wasm:
	$(ENGINE_DIR)/wasm/build.sh $(WASM_OUT_DIR)
	$(NODE) $(ENGINE_DIR)/wasm/smoke.mjs $(WASM_OUT_DIR)/earmark.wasm

# The runner refuses mislabelled runs (random weights outside --split smoke, a test split
# without a frozen threshold, ...) and exits 2; see `--help` for every option.
eval-dev:
	@if [ -z "$(strip $(EVAL_ARGS))" ]; then \
	  echo 'usage: make eval-dev EVAL_ARGS="--config M --checkpoint CKPT --manifest DEV/manifest.parquet [--embeddings EMB.npz] [--check]"' >&2; \
	  echo '       make eval-dev EVAL_ARGS=--help   lists every option' >&2; \
	  exit 2; \
	fi
	@scripts/disk_guard.sh
	PYTHONPATH=python $(PYTHON) -m $(DEV_RUNNER) $(EVAL_ARGS)

disk-guard:
	@scripts/disk_guard.sh

clean-caches:
	rm -rf $(CACHE_DIRS) .pytest_cache
	find python tests contract -type d -name __pycache__ -prune -exec rm -rf {} +
	@scripts/disk_guard.sh || true
