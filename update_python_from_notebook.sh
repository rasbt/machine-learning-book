#!/bin/bash

for ch in ch[0-9][0-9] ; do
  pushd "${ch}"
  for ipynb in *.ipynb ; do
    py="${ipynb/ipynb/py}"
    if [ -e "${py}" ]; then
      python ../.convert_notebook_to_script.py --input "${ipynb}" --output "${py}"
    fi
  done
  popd
done
