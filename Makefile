# Earmark developer entry points. Works with GNU make 3.81 (macOS) and newer.
# Override any variable on the command line, e.g. `make test PYTHON=python3`.

PYTHON ?= .venv/bin/python
CMAKE ?= $(if $(wildcard .venv/bin/cmake),.venv/bin/cmake,cmake)
CTEST ?= $(if $(wildcard .venv/bin/ctest),.venv/bin/ctest,ctest)
NINJA ?= $(if $(wildcard .venv/bin/ninja),$(abspath .venv/bin/ninja),ninja)

ENGINE_DIR ?= engine
ENGINE_BUILD_DIR ?= engine/build
ENGINE_BUILD_TYPE ?= RelWithDebInfo
ENGINE_CMAKE_ARGS ?=

# Module run by `make eval-dev` once the PyTorch-stream dev runner exists.
DEV_RUNNER ?= earmark.eval.dev_runner
DEV_RUNNER_FILE = python/$(subst .,/,$(DEV_RUNNER)).py
EVAL_ARGS ?=

PYTEST_ARGS ?=
CACHE_DIRS = .cache/huggingface .cache/torch .cache/uv .cache/pip .cache/npm .cache/ms-playwright

.PHONY: help test gates codegen contract-check engine eval-dev disk-guard clean-caches

help:
	@echo "make test            CPU unit tests (excludes the slow and gate markers)"
	@echo "make gates           acceptance gates (needs fetched eval data)"
	@echo "make codegen         regenerate constants from contract/signal.yaml"
	@echo "make contract-check  fail if the generated constants are stale"
	@echo "make engine          configure, build and ctest the C++ engine (Ninja)"
	@echo "make eval-dev        score the dev split with the PyTorch-stream runner"
	@echo "make disk-guard      fail when less than 5 GiB is free"
	@echo "make clean-caches    delete ./.cache downloads and Python caches"

test:
	CUDA_VISIBLE_DEVICES= $(PYTHON) -m pytest -q -m "not slow and not gate" $(PYTEST_ARGS)

gates:
	$(PYTHON) -m pytest -q -m gate $(PYTEST_ARGS)

codegen:
	$(PYTHON) contract/codegen.py

contract-check:
	$(PYTHON) contract/codegen.py --check

engine:
	@test -f $(ENGINE_DIR)/CMakeLists.txt || { echo "make engine: $(ENGINE_DIR)/CMakeLists.txt does not exist yet" >&2; exit 1; }
	$(CMAKE) -S $(ENGINE_DIR) -B $(ENGINE_BUILD_DIR) -G Ninja -DCMAKE_MAKE_PROGRAM=$(NINJA) -DCMAKE_BUILD_TYPE=$(ENGINE_BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $(ENGINE_CMAKE_ARGS)
	$(CMAKE) --build $(ENGINE_BUILD_DIR)
	$(CTEST) --test-dir $(ENGINE_BUILD_DIR) --output-on-failure

eval-dev: disk-guard
	@test -f $(DEV_RUNNER_FILE) || { echo "make eval-dev: $(DEV_RUNNER_FILE) does not exist yet (placeholder target)" >&2; exit 1; }
	PYTHONPATH=python $(PYTHON) -m $(DEV_RUNNER) $(EVAL_ARGS)

disk-guard:
	@scripts/disk_guard.sh

clean-caches:
	rm -rf $(CACHE_DIRS) .pytest_cache
	find python tests contract -type d -name __pycache__ -prune -exec rm -rf {} +
	@scripts/disk_guard.sh || true
