import pandas as pd
import numpy as np
import os, re, json, ast
from collections import defaultdict,Counter
from multiprocessing.dummy import Pool
from string import ascii_uppercase
from string import ascii_lowercase
import subprocess, sys, getopt, time
import math,copy
import linecache, itertools
from itertools import combinations
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib, requests
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
from Bio import Align
from Bio.Align import substitution_matrices


###将整数列表转换为区间
def convert_nums_to_intervals(num_list):
    for a, b in itertools.groupby(enumerate(num_list), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield [b[0][1], b[-1][1]]

###将区间转换为整数列表
def convert_intervals_to_nums(intervals):
    # intervals = [[9, 279], [357, 369], [393, 494]]
    num_list = []
    for interval in intervals:
        nums = list(range(interval[0],interval[1]+1))
        num_list.extend(nums)
    return num_list

###两个区间列表之间取交集
def overlap_intervals(intervals1,intervals2):
    # intervals1,intervals2 = [[11, 300],[320,450]], [[9, 279], [357, 369], [393, 494]]
    num_list1 = convert_intervals_to_nums(intervals1)
    num_list2 = convert_intervals_to_nums(intervals2)
    overlap_nums = [i for i in num_list1 if i in num_list2]
    overlap_intervals = list(convert_nums_to_intervals(overlap_nums))
    return overlap_intervals

###合并区间
def union_ranges(range_list):
    ##range_list = [[1,3],[4,7],[7,9],[8,10]]
    union_range = []
    for begin,end in sorted(range_list):
        if union_range and union_range[-1][1] >= begin - 1:
            union_range[-1][1] = max(union_range[-1][1], end)
        else:
            union_range.append([begin,end])
    return union_range



###双序列比对，获取比对上的片段
class PairwiseSeqAlign():
    def __init__(self):
        self.aligner = Align.PairwiseAligner()
        self.aligner.mode = 'global'
        self.aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
        self.aligner.open_gap_score = -10
        self.aligner.extend_gap_score = -0.5

    def extract_align_ranges(self,seq1,seq2):
        self.seq1,self.seq2 = seq1,seq2
        need_alignment = self.aligner.align(self.seq1, self.seq2)[0]
        m1,n1 = need_alignment.path[0]
        seg1_list,seg2_list = [],[]
        for m2,n2 in need_alignment.path[1:]:
            if (m2 > m1)&(n2 > n1):
                seg1 = [m1+1,m2]
                seg2 = [n1+1,n2]
                seg1_list.append(seg1)
                seg2_list.append(seg2)
            m1,n1 = m2,n2
        return union_ranges(seg1_list),union_ranges(seg2_list)


###uniprot


###ebi api
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


def obtain_aligned_ranges(len_unp_range,len_pdb_range,sequence,unp_range,pdb_id,chain_id,pdb_range):
    # row = aabb1.iloc[0]
    # len_unp_range,len_pdb_range,sequence,unp_range,pdb_id,chain_id,pdb_range = row['len_unp_range'],row['len_pdb_range'],row['sequence'],row['unp_range'],row['pdb_id'],row['chain_id'],row['pdb_range']
    unp_range,pdb_range = json.loads(unp_range),json.loads(pdb_range)
    range_diff = len_unp_range-len_pdb_range
    ###根据api提供的范围获取unp的序列和索引
    unp_range_seq,unp_range_indices = sequence[unp_range[0]-1:unp_range[1]],list(range(unp_range[0],unp_range[1]+1))
    ###根据api提供的范围获取pdb的序列和索引
    pdb_residues = pd.read_csv('/data/jiangxin/tools/mapping/pdb_infos/pdb_residues_'+pdb_id,sep='\t').sort_values(by=['struct_asym_id','residue_number'])
    pdb_molecules = pd.read_csv('/data/jiangxin/tools/mapping/pdb_infos/pdb_molecules_'+pdb_id,sep='\t')
    othermols = pdb_molecules[pdb_molecules['molecule_type']!='polypeptide(L)']
    othermols_infos = [(row['chain_id'],row['struct_asym_id']) for index,row in othermols[['chain_id','struct_asym_id']].iterrows()]
    pdb_protein_residues = pdb_residues[pdb_residues.apply(lambda x: (x['chain_id'],x['struct_asym_id']) not in othermols_infos,axis=1)]#去除其他化合物的信息
    chain_range_residues = pdb_protein_residues[(pdb_protein_residues['chain_id']==chain_id)&
                                                (pdb_protein_residues['residue_number']>=pdb_range[0])&
                                                (pdb_protein_residues['residue_number']<=pdb_range[1])]
    pdb_range_seq,pdb_range_indices = ''.join(list(chain_range_residues['single_letter'])),list(chain_range_residues['residue_number'])
    ###序列比对，获取比对上的序列的索引区间
    PSAligner = PairwiseSeqAlign()
    unp_aligned_intervals,pdb_aligned_intervals = PSAligner.extract_align_ranges(unp_range_seq,pdb_range_seq)
    ###根据比对上的序列索引区间，获取各自原始序列的范围
    unp_aligned_range = [[unp_range_indices[j-1] for j in i] for i in unp_aligned_intervals]
    pdb_aligned_range = [[pdb_range_indices[j-1] for j in i] for i in pdb_aligned_intervals]
    aligned_len = len(convert_intervals_to_nums(pdb_aligned_range))
    ###获取有无坐标残基的pdb序列范围
    obs_range_residues = pdb_protein_residues[(pdb_protein_residues['chain_id']==chain_id)&(pdb_protein_residues['observed_ratio']>0)]
    pdb_obs_range = list(convert_nums_to_intervals(list(obs_range_residues['residue_number'])))
    pdb_missing_residues = pdb_protein_residues[(pdb_protein_residues['chain_id']==chain_id)&(pdb_protein_residues['observed_ratio']==0)]
    pdb_missing_range = list(convert_nums_to_intervals(list(pdb_missing_residues['residue_number'])))
    ###与pdb比对上的范围合并，并对应到unp上
    pdb_aligned_obs_range = overlap_intervals(pdb_aligned_range,pdb_obs_range)
    pdb_unp_aligned_dict = dict(zip(convert_intervals_to_nums(pdb_aligned_range),convert_intervals_to_nums(unp_aligned_range)))
    unp_aligned_obs_residues = [pdb_unp_aligned_dict[i] for i in convert_intervals_to_nums(pdb_aligned_obs_range)]
    unp_aligned_obs_range = list(convert_nums_to_intervals(unp_aligned_obs_residues))
    aligned_obs_len = len(unp_aligned_obs_residues)
    return range_diff,unp_aligned_range,pdb_aligned_range,aligned_len,pdb_obs_range,pdb_missing_range,unp_aligned_obs_range,pdb_aligned_obs_range,aligned_obs_len


# aabb1[['range_diff','unp_aligned_range','pdb_aligned_range','aligned_len','pdb_obs_range','pdb_missing_range','unp_aligned_obs_range',
#        'pdb_aligned_obs_range','aligned_obs_len']] = aabb1.apply(lambda x: 
#                                                obtain_aligned_ranges(x['len_unp_range'],x['len_pdb_range'],x['sequence'],x['unp_range'],
#                                                                      x['pdb_id'],x['chain_id'],x['pdb_range']),axis=1,result_type='expand')

def map_uniprot_to_bestpdb(unp_id):###coverage,resolution
    best_response = requests.get('https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/%s'%(unp_id))
    if best_response.text!='{}':
        best_result = pd.DataFrame(json.loads(best_response.text)[unp_id])
        best_result['unp_id'] = unp_id
        return best_result
    else:
        return pd.DataFrame()


