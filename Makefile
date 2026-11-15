# Makefile for TerrariaBench Junction Submission

PYTHON ?= python

.PHONY: all eval test-anti-cheat serve-local-demo deploy-modal train-grpo clean

all: test-anti-cheat eval

# Runs full evaluation across seeds 7, 11, and 13, generating frames, checkpoints, EVAL_RESULTS.md, and docs/index.html
eval:
	$(PYTHON) reporting/generate_eval_report.py

# Runs the 5-point reward integrity and anti-cheat test suite
test-anti-cheat:
	$(PYTHON) -m pytest tests/test_reward_integrity.py -v

# Serves the live interactive dashboard locally on port 8080
serve-local-demo:
	$(PYTHON) cloud/modal_app.py --local 8080

# Deploys live demo to Modal (public https://...modal.run URL)
deploy-modal:
	$(PYTHON) -m modal deploy cloud/modal_app.py

# Launches H100 GRPO post-training on Modal
train-grpo:
	$(PYTHON) -m modal run cloud/modal_app.py::train_grpo

# Auto-plays 24-step trajectory and exports MP4 video (sim mode)
record-video-sim:
	$(PYTHON) scripts/auto_play_and_record.py --mode sim

# Auto-plays inside the live Terraria window and exports MP4 video (live mode)
record-video-live:
	$(PYTHON) scripts/auto_play_and_record.py --mode live

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
