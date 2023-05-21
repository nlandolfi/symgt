precommit:
	make fmt
	make lint
	make types
	make test
build:
	python3 -m build
deploy:
	python3 -m twine upload dist/*
install:
	pip install -e .
fmt:
	python -m black .
lint:
	python -m ruff .
types:
	mypy src
test:
	python tests/01_smoke_test:_IIDModel.py
	python tests/02_smoke_test:_ExchangeableModel.py
	python tests/03_smoke_test:_algorithms.py
	python tests/04_smoke_test:_utils.py
	python tests/05_smoke_test:_golden.py
	python tests/06_smoke_test:_golden2.py
freeze:
	pip freeze > requirements.txt

.PHONY: setupenv
setupenv:
	python3 -m venv symgt_env
	. symgt_env/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	make install
