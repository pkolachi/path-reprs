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

def verb_transitive(verbid, sent):
    nomnum = len([y for y in [x for x in sent if x['head'] == verbid] if y['upostag'] == 'NOUN'])
    if nomnum == 1:
        return [1,0,0,0]
    elif nomnum == 2:
        return [0,1,0,0]
    elif nomnum == 3:
        return [0,0,1,0]
    else:
        return [0,0,0,1]
    
def verb_argument_count(verbid, sent):
    return len([x for x in sent if x['head'] == verbid])

def verb_argument_types(verbid, sent):
    children = [x for x in sent if x['head'] == verbid]
    argtypes = []
    for c in children:
        argtypes.append(c['upostag'])

    return set(argtypes), len(set(argtypes))

def verb_argument(verbid, sent):
    return 1 if len([y for y in [x for x in sent if x['head'] == verbid] if y['upostag'] == 'VERB']) > 0 else 0

def verb_depth(verbline, sent):
    depth = 0

    if verbline['head'] != 0 and isinstance(verbline['head'], int):
        head = verbline['head']
        while head != 0:
            depth += 1
            headline = [x for x in sent if x['id'] == head][0]
            head = headline['head']

    return depth

def verb_head_type(verbline, sent):
    if verbline['head'] != 0 and isinstance(verbline['head'], int):
        return [x for x in sent if x['id'] == verbline['head']][0]['upostag']
    else:
        return 'None'

def verb_root(verbline):
    return 1 if verbline['head'] == 0 else 0

def verb_distance(sent):
    dist = []
    verblines = [x for x in sent if x['upostag'] == 'VERB']
    for v1,v2 in zip(verblines, verblines[1:]):
        dist.append(abs(v1['id'] - v2['id']))

    return sum(dist) / len(dist) if len(dist) > 0 else -1

def verb_number(sent):
    return len([x for x in sent if x['upostag'] == 'VERB'])
    
def verb_features(verbline, sent):
    feat = []
    #feat.extend((verbline['lemma'], str(verbline['id']), str(len(sent)), str(len(verbline['lemma']))))
    feat.append(verbline['lemma'])
    feat.append(verb_argument_count(verbline['id'],sent))
    feat.append(verb_argument_types(verbline['id'],sent)[1])
    #feat.extend(verb_argument_types(verbline['id'],sent)[0])
    feat.extend(verb_transitive(verbline['id'],sent))
    feat.append(verb_argument(verbline['id'],sent))
    feat.append(verb_root(verbline))
    feat.append(verb_depth(verbline, sent))
    #feat.append(verb_head_type(verbline, sent))
    feat.append(verb_number(sent))
    feat.append(verbline['id'])
    feat.append(len(sent))
    #feat.append(str(verb_distance(sent)))
        
    return feat    

def extract_verbs(corpusfile):
    verbs = []
    udsents = udparse(io.open(corpusfile, 'r', encoding='utf-8').read())
    for sent in udsents:
        for s in sent:
            if s['upostag'] == 'VERB':
                verbs.append(verb_features(s, sent))
    
    return verbs
                 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="Categories to use", default="true")
    parser.add_argument("-f", "--outfile", help="vectorizer to use to extract features", default="count")
    args = parser.parse_args()
    verbs = pd.DataFrame(extract_verbs(args.infile))
    #verbs.to_csv(args.outfile,sep=',',index=False,header=False)
    
if __name__ == "__main__":
    main()
