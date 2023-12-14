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
	python tests/07_smoke_test:_subset_symmetry_utils.py
	python tests/08_smoke_test:_generalized_algorithm.py
	python tests/09_smoke_test:_IndependentSubpopulationsModel.py
	python tests/10_smoke_test:_SubsetSymmetryModel.py
	python tests/11_smoke_test:_U_from_q_orbits.py
	python tests/12_smoke_test:_symmetric_orbit_multfn.py
freeze:
	pip freeze > requirements.txt

.PHONY: setupenv
setupenv:
	python3 -m venv symgt_env
	. symgt_env/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	make install
