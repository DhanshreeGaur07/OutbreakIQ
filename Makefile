.PHONY: setup data features train predict all demo dashboard test lint docker
setup:        ## create venv and install pinned dependencies
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
data:
	python main.py ingest && python main.py preprocess
features:
	python main.py features
train:
	python main.py train
predict:
	python main.py predict
all:          ## full pipeline on real JHU + OWID data
	python main.py all
demo:         ## offline end-to-end run on simulated data
	python main.py all --synthetic
dashboard:
	streamlit run app/dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker compose build
