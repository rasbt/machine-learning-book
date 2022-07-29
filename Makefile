install:
	@export PYENV_VERSION=3.8.10
	@pip install --upgrade pip
	@pip install poetry==1.1.12
	@poetry update
	@poetry shell