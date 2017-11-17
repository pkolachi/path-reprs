#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Sudheer Kolachina"
__copyright__ = "Copyright 2017, The Bullshit project"
__credits__ = ["mufula tufula inspirators"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Klassic Koala"
__email__ = "koala@mit.edu"
__status__ = "Pregnancy"

import argparse
import io
import os
import pandas as pd
import re
from conllu.parser import parse as udparse

def path_to_string(sentpath):
    strings = []
    for s in sentpath:
        strings.append(' '.join([x[1] for x in reversed(s)]))
    
    eliminate_substrings(strings)        
    return strings

def eliminate_substrings(strings):
    for (s1,s2) in [(p,q) for p in strings for q in strings if p != q]:
            if s1.strip() in s2.strip():
                try:
                    strings.remove(s1)
                except ValueError:
                    pass

    return strings
        
def extract_paths(sent):
    sentpath = []
    for s in sent:
        rootpath = []
        if s['head'] != 0 and isinstance(s['head'], int):
            head = s['head']
            while head != 0:
                headline = [x for x in sent if x['id'] == head][0]
                head = headline['head']
                rootpath.append((head,headline['form']))
            rootpath.insert(0,(s['id'], s['form']))
            sentpath.append(rootpath)
        else:
            sentpath.append([(s['id'],s['form']), ('', '')])
    
    return sentpath       

def extract_pathstrings(corpusfile):
    paths = []
    udsents = udparse(io.open(corpusfile, 'r', encoding='utf-8').read())
    for sent in udsents:
        sentpath = extract_paths(sent)
        paths.append(path_to_string(sentpath))        
        #paths.append(extract_paths(sent))        

    return paths
                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="treebank file")
    parser.add_argument("-o", "--outfile", help="output file")
    args = parser.parse_args()
    paths = extract_pathstrings(args.infile)
    with io.open(args.outfile,'w',encoding='utf-8') as f:
        for p in paths:
            f.write('\n'.join(p)+'\n')
    f.close()

if __name__ == "__main__":
    main()
