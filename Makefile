install:
	# Used with conda 4.13.0
	conda config --append channels conda-forge
	conda install python=3.8.10 numpy=1.21.2 scipy=1.7.0 matplotlib=3.4.3 scikit-learn=1.0 pandas=1.3.2
create:
	conda env create environment.yml --force
	conda activate machine-learning-book
