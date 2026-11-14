PYTHON ?= python

.PHONY: install calibrate eval test test-exploits train-gemma dashboard console bridge

install:
	$(PYTHON) -m pip install -e ".[dev,agents]"

calibrate:
	$(PYTHON) -m terraria_bench.calibrate

eval:
	$(PYTHON) -m terraria_bench.harness.runner --eval-standard

test:
	$(PYTHON) -m pytest tests/ -v

test-exploits:
	$(PYTHON) -m pytest tests/test_reward_exploits.py -v

train-gemma:
	$(PYTHON) -m terraria_bench.harness.train_grpo

dashboard:
	$(PYTHON) -m uvicorn dashboard.app:app --host 127.0.0.1 --port 8000

console:
	$(PYTHON) -m terraria_bench.web_console --task first-night --seed 7 --port 8080

bridge:
	$(PYTHON) -m terraria_bench.human_recorder --serve --host 127.0.0.1 --port 8765
