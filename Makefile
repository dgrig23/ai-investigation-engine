PY := .venv/bin/python

.PHONY: install test lint corpus hotpotqa index-incidents index-incidents-fixed index-hotpotqa serve eval-retrieval eval-answers report

install:
	uv venv --python 3.11 .venv && uv pip install --python $(PY) -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

corpus:            ## generate the synthetic Acme Play corpus
	$(PY) -m scripts.generate_incident_corpus

hotpotqa:          ## download + build the HotpotQA subset and evaluation questions
	$(PY) -m scripts.prepare_hotpotqa

index-incidents:
	$(PY) -m scripts.index_corpus --corpus incidents

index-incidents-fixed:
	$(PY) -m scripts.index_corpus --corpus incidents --strategy fixed

index-hotpotqa:
	$(PY) -m scripts.index_corpus --corpus hotpotqa

serve:
	$(PY) -m uvicorn app.api.main:app --reload --port 8000

eval-retrieval:
	$(PY) -m eval.run_eval retrieval --dataset investigation
	$(PY) -m eval.run_eval retrieval --dataset investigation --strategy fixed
	$(PY) -m eval.run_eval retrieval --dataset hotpotqa --limit 300

eval-answers:
	$(PY) -m eval.run_eval answers --dataset investigation --mode agent
	$(PY) -m eval.run_eval answers --dataset investigation --mode single_shot

report:
	$(PY) -m eval.report
