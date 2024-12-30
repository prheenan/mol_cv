#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

function run(){
  conda env create --force --yes -f env.yaml
  conda activate mol_cv
  rm -rf lib/
  mkdir -p lib/
  # download the various data sets
  git clone https://github.com/IanAWatson/Lilly-Medchem-Rules lib/Lilly-Medchem-Rules
  cd lib/Lilly-Medchem-Rules/ > /dev/null
  make
  cd - > /dev/null
  git clone https://github.com/IUPAC/Dissociation-Constants lib/Dissociation-Constants
  # RTlogD has a bunch of other stuff we don't care about; just download the data
  git clone -n --depth=1 --filter=tree:0 https://github.com/WangYitian123/RTlogD lib/RTlogD
  cd lib/RTlogD > /dev/null
  git sparse-checkout set --no-cone /original_data
  git checkout
  cd - > /dev/null
}

run
