precommit:
	make fmt
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
types:
	mypy src
test:
	python tests/01_smoke_test:_IIDModel.py
freeze:
	pip freeze > requirements.txt

.PHONY: setupenv
setupenv:
	python3 -m venv env
	. env/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	make install
