create_data_folders:
	mkdir -p data/00_raw
	mkdir -p data/01_processed
	mkdir -p data/02_sparse_model
	mkdir -p data/03_dense_model
	mkdir -p data/04_results
	mkdir -p external_dependencies

install_external_deps_m1: 
	. scripts/install_m1.sh

install_requirements_conda:
	conda install --file requirements.txt
	conda install black flake8 isort mypy pre-commit

install_requirements_pip:
	pip install -r requirements.txt
	pip install black flake8 isort mypy pre-commit

install_precommit:
	pre-commit install --hook-type pre-commit

install_m1: create_data_folders install_external_deps_m1 install_requirements_conda

install_conda: install_requirements_conda install_precommit

install_pip: install_requirements_pip install_precommit