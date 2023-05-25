#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


map_3_to_1 = {"GLY": "G", "ALA": "A", "SER": "S", "THR": "T", "CYS": "C",
              "VAL": "V", "LEU": "L", "ILE": "I", "MET": "M", "PRO": "P",
              "PHE": "F", "TYR": "Y", "TRP": "W", "ASP": "D", "GLU": "E",
              "ASN": "N", "GLN": "Q", "HIS": "H", "LYS": "K", "ARG": "R"}

# In[ ]:


parser = argparse.ArgumentParser(description='This script still needs to be improved.(Jing Xin)')
parser.add_argument('--input', '-i', type=str, required=True, help='PDB ID, like: 1a2k')
parser.add_argument('--path', '-d', type=str, default='/data/jiangxin/tools/mapping/pdb_infos/', required=False, help='The path to save results')

args = parser.parse_args()

# In[3]:


def get_pdb_summary(pdb_id):
    # pdb_id = '6c9j'
    pdb_summary_response = requests.get('https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/%s'%(pdb_id))
    summary_raw_result = json.loads(pdb_summary_response.text)
    summary_need = {'title':[],'experimental_method':[],'number_of_entities':[],'assemblies':[]}
    for summary_data in summary_raw_result[pdb_id]:
        for key in list(summary_need):
            if key in summary_data:
                summary_need[key].append(summary_data[key])
            else:
                summary_need[key].append(np.nan)
    pdb_summary = pd.DataFrame(summary_need).explode('experimental_method')
    pdb_summary['pdb_id'] = pdb_id
    if pdb_summary.shape[0]!=1:
        print('This pdb_id has multiple summary, need check: %s'%(pdb_id))
    return pdb_summary


# In[4]:


def get_pdb_molecules(pdb_id):
    pdb_molecule_response = requests.get('https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/%s'%(pdb_id))
    molecule_raw_result = json.loads(pdb_molecule_response.text)
    ###注意此处仅仅获取了多肽链的信息，小分子、核酸信息不全，没有例子，不知道提哪些
    molecule_need = {'entity_id':[],'molecule_type':[],'mutation_flag':[],'molecule_name':[],'sequence':[],'pdb_sequence':[],'in_chains':[],
                     'pdb_sequence_indices_with_multiple_residues':[],'in_struct_asyms':[],'gene_name':[],'chem_comp_ids':[]}
    for entity_data in molecule_raw_result[pdb_id]:
        for key in molecule_need:
            if key in entity_data:
                molecule_need[key].append(entity_data[key])
            else:
                molecule_need[key].append(np.nan)
    pdb_molecule = pd.DataFrame(molecule_need).rename(columns={'sequence':'raw_pdb_sequence','pdb_sequence':'real_pdb_sequence'})
    pdb_molecule['pdb_id'] = pdb_id
    pdb_molecules = pdb_molecule.explode(['in_chains','in_struct_asyms']).rename(columns={'in_chains':'chain_id',
                                                                                          'in_struct_asyms':'struct_asym_id'})
    return pdb_molecules


# In[5]:


def get_pdb_residues(pdb_id):
    # pdb_id = '6c9j'
    pdb_residues_response = requests.get('https://www.ebi.ac.uk/pdbe/api/pdb/entry/residue_listing/%s'%(pdb_id))
    residues_raw_result = json.loads(pdb_residues_response.text)
    chain_residues_list = []
    for entity_data in residues_raw_result[pdb_id]['molecules']:
        for chain_data in entity_data['chains']:
            chain_residues = pd.DataFrame(chain_data['residues'])
            chain_residues['struct_asym_id'] = chain_data['struct_asym_id']
            chain_residues['chain_id'] = chain_data['chain_id']
            chain_residues_list.append(chain_residues)
    pdb_residues = pd.concat(chain_residues_list,axis=0).reset_index(drop=1)
    pdb_residues['pdb_id'] = pdb_id
    pdb_residues['single_letter'] = pdb_residues.apply(lambda x: map_3_to_1[x['residue_name']] if x['residue_name'] in map_3_to_1 else 'X',axis=1)
    pdb_residues['res'] = pdb_residues['single_letter']+pdb_residues['author_residue_number'].astype(str)+pdb_residues['author_insertion_code'].fillna('')
    return pdb_residues



def get_pdb_experiment(pdb_id):
    pdb_experiment_response = requests.get('https://www.ebi.ac.uk/pdbe/api/pdb/entry/experiment/%s'%(pdb_id))
    experiment_raw_result = json.loads(pdb_experiment_response.text)
    experiment_infos = {'experimental_method':[],'experimental_method_class':[],'resolution':[],
                        'resolution_high':[],'resolution_low':[],'r_factor':[],'r_free':[]}
    for data in experiment_raw_result[pdb_id]:
        for key in experiment_infos:
            if key in data:
                experiment_infos[key].append(data[key])
            else:
                experiment_infos[key].append(np.nan)
    pdb_experiment = pd.DataFrame(experiment_infos)
    return pdb_experiment


# In[ ]:


pdb_id = args.input
save_dir = args.path
pdb_summary = get_pdb_summary(pdb_id)
pdb_molecules = get_pdb_molecules(pdb_id)
pdb_residues = get_pdb_residues(pdb_id)
pdb_summary.to_csv(save_dir+'pdb_summary_'+pdb_id,sep='\t',index=0)
pdb_molecules.to_csv(save_dir+'pdb_molecules_'+pdb_id,sep='\t',index=0)
pdb_residues.to_csv(save_dir+'pdb_residues_'+pdb_id,sep='\t',index=0)
time.sleep(5)###别太频繁，小心被封


# In[ ]:




