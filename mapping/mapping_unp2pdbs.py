#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os, re, json, ast
from collections import defaultdict,Counter
from multiprocessing.dummy import Pool
from string import ascii_uppercase
from string import ascii_lowercase
import subprocess, sys, getopt, time
import math,copy
import linecache
from itertools import combinations
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib, requests
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
import argparse
import warnings
warnings.filterwarnings('ignore')


# In[9]:

parser = argparse.ArgumentParser(description='This script still needs to be improved.(Jing Xin)')
parser.add_argument('--input', '-i', type=str, required=True, help='UniProt ID, like: P00519')
parser.add_argument('--path', '-d', type=str, default='/data/jiangxin/tools/mapping/unp2pdbs/', required=False, help='The path to save results')

args = parser.parse_args()


# In[7]:


def get_uniprot_infos(unp_id):
    # unp_id = 'Q13131'
    ###获取uniprot id的序列信息,这里只提取了部分
    unp_seq_response = requests.get('https://www.ebi.ac.uk/proteins/api/proteins/%s'%(unp_id))
    unp_raw_result = json.loads(unp_seq_response.text)
    unp_id = unp_raw_result['accession']
    entry = unp_raw_result['id']
    sequence = unp_raw_result['sequence']['sequence']
    uniprot_infos = pd.DataFrame({'unp_id':[unp_id],'entry':[entry],'sequence':[sequence]})
    return uniprot_infos


# In[8]:


def map_uniprot_to_pdb(unp_id):
    # unp_id = 'Q13131'##Q2M2I8
    unp2pdbs_response = requests.get('https://www.ebi.ac.uk/pdbe/api/mappings/all_isoforms/%s'%(unp_id))
    unp2pdbs_raw_result = json.loads(unp2pdbs_response.text)[unp_id]['PDB']
    unp2pdbs_dict = {'pdb_id':[],'entity_id':[],'chain_id':[],'is_canonical':[],'identity':[],'unp_range':[],'pdb_range':[]}
    for pdb in list(unp2pdbs_raw_result):
        for pdb_data in unp2pdbs_raw_result[pdb]:###insdel
            unp2pdbs_dict['pdb_id'].append(pdb)
            for key in ['entity_id','chain_id','is_canonical','identity']:
                unp2pdbs_dict[key].append(pdb_data[key])
            unp_range = [pdb_data['unp_start'],pdb_data['unp_end']]
            pdb_range = [pdb_data['start']['residue_number'],pdb_data['end']['residue_number']]
            unp2pdbs_dict['unp_range'].append(unp_range)
            unp2pdbs_dict['pdb_range'].append(pdb_range)
    unp2pdbs_result = pd.DataFrame(unp2pdbs_dict)
    unp2pdbs_result['unp_id'] = unp_id
    unp2pdbs_result['len_unp_range'] = unp2pdbs_result.apply(lambda x: x['unp_range'][1]-x['unp_range'][0]+1,axis=1)
    unp2pdbs_result['len_pdb_range'] = unp2pdbs_result.apply(lambda x: x['pdb_range'][1]-x['pdb_range'][0]+1,axis=1)
    return unp2pdbs_result


# In[ ]:


unp_id = args.input
save_dir = args.path
uniprot_infos = get_uniprot_infos(unp_id)
unp2pdbs_result = map_uniprot_to_pdb(unp_id)
uniprot_infos.to_csv(save_dir+'unp_infos_'+unp_id,sep='\t',index=0)
unp2pdbs_result.to_csv(save_dir+'unp2pdbs_'+unp_id,sep='\t',index=0)
time.sleep(5)###别太频繁，小心被封



