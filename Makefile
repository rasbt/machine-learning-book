install:
	# Used with conda 4.13.0
	conda config --append channels conda-forge
	conda env update -n pyml-book --file environment.yml --prune
create:
	# Required to remove old dependencies due to broken prune after conda 4.4
	# https://github.com/conda/conda/issues/7279
	conda env create environment.yml --force