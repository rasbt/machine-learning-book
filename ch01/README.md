Python Machine Learning - Code Examples


##  Chapter 1: Giving Computers the Ability to Learn from Data


---



## Setting Up Your Python Environment

This chapter does not contain any code examples, but we recommend you to set up and check your Python before your proceed with the next chapters. 

For more detailed setup instructions, please refer to the section ***Installing Python and packages from the Python Package Index*** in Chapter 1.


## Checking Your Python Environment

To verify that your Python environment is set up for the following chapters, we recommend running the `[../python-environment-check.py](../python-environment-check.py)` script provided in the main folder of this repository. 

You can run the `python-environment-check.py` via 

    python python-environment-check.py

or 

    python3 python-environment-check.py

Shown below is an example output:

```python
(base) sebastian@MacBook-Air ~/Desktop/Python-Machine-Learning-PyTorch-Edition/ch01 % python ../python-environment-check.py 
[OK] Your Python version is 3.9.6 | packaged by conda-forge | (default, Jul 11 2021, 03:35:11) 
[Clang 11.1.0 ]
[OK] numpy 1.21.2
[OK] scipy 1.7.0
[OK] matplotlib 3.4.3
[OK] sklearn 1.0
[OK] pandas 1.3.2
```


## Jupyter Notebooks

Some readers were wondering about the .ipynb of the code files contained in this repository -- these files are Jupyter notebooks (formerly known as IPython notebooks). 

Compared to regular .py scripts, Jupyter notebooks allow us to have everything in one place:

- Our code.
- The results from executing the code.
- Plots of our data.
- Documentation supporting the handy Markdown and LaTeX syntax for typing and rendering mathematical notation.

Please see the https://jupyter.org/install website for the latest installation instructions.

Two official applications can open Jupyter notebooks: the original Jupyter Notebook app and the newer Jupyter Lab app (and VS Code has Jupyter notebook support, too). The notebooks provided in this repository are compatible with both.

Finally, please note that the Jupyter notebooks provided in this repository are optional, although we highly recommend them. All code examples found in this book are also available via .py script files (which were converted from the Jupyter notebooks to ensure that they contain the identical code.)
