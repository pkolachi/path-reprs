#!/bin/bash

# Commands used to extract several data files
# mostly included here for log purposes 

python extract_dependencypaths.py \
    -i ../lib/ud-treebanks-v2.0/UD_English/en-ud-train.conllu \
    -o ../data/en-ud-deppaths.in 

python extract_dependencypaths.py \
    -i ../lib/ud-treebanks-v2.0/UD_Hindi/hi-ud-train.conllu \
    -o ../data/hi-ud-deppaths.in 

