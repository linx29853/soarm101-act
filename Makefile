SHELL := /bin/bash
PYTHON ?= python3
CONFIG ?= configs/act_so101.json
export PYTHONPATH := src:src/lerobot/src

.PHONY: help setup check test record train train-smoke rollout clean

help:
	@printf '%s\n' \
		'make setup       Install LeRobot, the SO-101 patch, and this project' \
		'make check       Check the Python environment, dataset, and ACT configuration' \
		'make test        Run dependency-free unit tests' \
		'make train-smoke Run one ACT forward/backward pass on synthetic data' \
		'make record      Record demonstrations; requires SOARM_* environment variables' \
		'make train       Train ACT with CONFIG=configs/act_so101.json' \
		'make rollout     Run a trained policy on SO-ARM101'

setup:
	bash scripts/bootstrap.sh --install

check:
	$(PYTHON) -m soarm101_act.cli check --config $(CONFIG)

test:
	$(PYTHON) -m unittest discover -s tests -v

train-smoke:
	$(PYTHON) -m soarm101_act.cli train-smoke --config $(CONFIG)

record:
	$(PYTHON) -m soarm101_act.cli record --config $(CONFIG)

train:
	$(PYTHON) -m soarm101_act.cli train --config $(CONFIG)

rollout:
	$(PYTHON) -m soarm101_act.cli rollout --config $(CONFIG) --policy-path outputs/train/act_so101/checkpoints/last/pretrained_model

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find . -path ./src/lerobot -prune -o -name __pycache__ -type d -exec rm -rf {} +
