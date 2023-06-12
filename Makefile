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

install_m1: install_external_deps_m1 install_requirements_conda install_md

install_conda: install_requirements_conda install_precommit install_md

install_pip: install_requirements_pip install_precommit install_md

sync:
	rsync -auv --progress --exclude-from=.gitignore --exclude=data --exclude=external_dependencies . ${USR}@euler.ethz.ch:/cluster/home/${USR}/code/RefinedMegaDepth