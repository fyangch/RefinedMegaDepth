create_data_folders:
	mkdir -p data/00_images
	mkdir -p data/01_features
	mkdir -p data/02_matches
	mkdir -p data/03_sparse
	mkdir -p data/04_dense
	mkdir -p data/05_metrics
	mkdir -p data/06_results
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

install_md:
	pip install -e .

install_m1: create_data_folders install_external_deps_m1 install_requirements_conda install_md

install_conda: install_requirements_conda install_precommit install_md

install_pip: install_requirements_pip install_precommit install_md