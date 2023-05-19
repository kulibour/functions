import pandas as pd
import numpy as np
import os, sys, pickle, math
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, QED, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Avalon import pyAvalonTools
from joblib import Parallel, delayed
import seaborn as sns
from collections import Counter

def convert_kd_to_pkd(kd,nm_flag):
    if pd.isna(kd):
        return np.nan
    else:
        if nm_flag == True:###单位为nM
            pkd = -1*(math.log10(kd/1e+9))
            return pkd
        else:###单位为M
            pkd = -1*(math.log10(kd))
            return pkd

def convert_pkd_to_kd(pkd,nm_flag):
    if pd.isna(pkd):
        return np.nan
    else:
        if nm_flag == True:###单位为nM
            kd = 10**(pkd*-1)*1e+9
            return kd
        else:###单位为M
            kd = 10**(pkd*-1)
            return kd

class wrapper_standardize_smile(object):
    def __init__(self, method_name):
        self.method_name = method_name

    def __call__(self, smile):
        try:
            mol = Chem.MolFromSmiles(smile)
            #除去氢、金属原子, 标准化分子
            clean_mol = rdMolStandardize.Cleanup(mol)
            #仅保留主要片段作为分子
            clean_parent_mol = rdMolStandardize.FragmentParent(clean_mol)
            #尝试中性化处理分子
            uncharger = rdMolStandardize.Uncharger()
            clean_parent_uncharge_mol = uncharger.uncharge(clean_parent_mol)
            clean_smile = Chem.MolToSmiles(clean_parent_uncharge_mol)
            return clean_smile
        except Exception as e:
            return 'wrong'

class process_database:
    def __init__(self, raw_file, database_path='./enamine/', num_worker=5):
        self.raw_file = raw_file
        self.database_path = database_path
        self.num_worker = num_worker
        os.makedirs(self.database_path+'preprocess',exist_ok=True)
        os.makedirs(self.database_path+'plot',exist_ok=True)
        os.makedirs(self.database_path+'info',exist_ok=True)
        print('0. Loading database...')
        if not os.path.exists(self.database_path+'preprocess/raw_data.tsv'):
            self.delimiter = ',' if self.raw_file.endswith('.csv') else '\t'
            self.raw_data = pd.read_csv(self.raw_file,sep=self.delimiter)
            self.raw_data['unique_id'] = [f'mol_{i}' for i in range(self.raw_data.shape[0])]
            self.raw_data.to_csv(self.database_path+'preprocess/raw_data.tsv',sep='\t',index=0)
        else:
            self.raw_data = pd.read_csv(self.database_path+'preprocess/raw_data.tsv',sep='\t')

    def filter_wrong_smiles(self):
        print('1. Filtering wrong SMILE...')
        if not os.path.exists(self.database_path+'preprocess/valid_data.tsv'):
            raw_smile_list = list(self.raw_data['smiles'])
            clean_smile_list = Parallel(n_jobs=self.num_worker)(delayed(wrapper_standardize_smile('wrapper'))(i) for i in raw_smile_list)
            self.raw_data['clean_smiles'] = clean_smile_list
            self.valid_data = self.raw_data[self.raw_data['clean_smiles']!='wrong']
            print('The number of wrong smiles: ',self.raw_data.shape[0]-self.valid_data.shape[0])
            self.valid_data.to_csv(self.database_path+'preprocess/valid_data.tsv',sep='\t',index=0)
        else:
            self.valid_data = pd.read_csv(self.database_path+'preprocess/valid_data.tsv',sep='\t')

    def filter_duplicated_smiles(self):
        print('2. Filtering duplicated SMILES...')
        if not os.path.exists(self.database_path+'preprocess/valid_nri_data.tsv'):
            self.valid_nri_data = self.valid_data.drop_duplicates(['clean_smiles'])
            print('The number of duplicated smiles: ',self.valid_data.shape[0]-self.valid_nri_data.shape[0])
            self.valid_nri_data.to_csv(self.database_path+'preprocess/valid_nri_data.tsv',sep='\t',index=0)
        else:
            self.valid_nri_data = pd.read_csv(self.database_path+'preprocess/valid_nri_data.tsv',sep='\t')


class wrapper_MolToSmiles(object):
    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        return mol

class wrapper_RDKitFPGen(object):
    def __call__(self, mol):
        rdkit_fpgen = AllChem.GetRDKitFPGenerator()
        rdkit_fp = rdkit_fpgen.GetFingerprint(mol)
        rdkit_fp_array = np.array(rdkit_fp)
        return rdkit_fp_array

class wrapper_AtomPairFPGen(object):
    def __call__(self, mol):
        atompair_fpgen = AllChem.GetAtomPairGenerator()
        atompair_fp = atompair_fpgen.GetFingerprint(mol)
        atompair_fp_array = np.array(atompair_fp)
        return atompair_fp_array

class wrapper_TorsionFPGen(object):
    def __call__(self, mol):
        torsion_fpgen = AllChem.GetTopologicalTorsionGenerator()
        torsion_fp = torsion_fpgen.GetFingerprint(mol)
        torsion_fp_array = np.array(torsion_fp)
        return torsion_fp_array

class wrapper_MorganFPGen(object):
    def __call__(self, mol):
        morgan_fpgen = AllChem.GetMorganGenerator(radius=2)
        morgan_fp = morgan_fpgen.GetFingerprint(mol)
        morgan_fp_array = np.array(morgan_fp)
        return morgan_fp_array

class wrapper_AvalonFPGen(object):
    def __call__(self, mol):
        avalon_fp = pyAvalonTools.GetAvalonFP(mol)
        avalon_fp_array = np.array(avalon_fp)
        return avalon_fp_array

class wrapper_LayeredFPGen(object):
    def __call__(self, mol):
        layered_fp = Chem.rdmolops.LayeredFingerprint(mol)
        layered_fp_array = np.array(layered_fp)
        return layered_fp_array

class wrapper_PatternFPGen(object):
    def __call__(self, mol):
        pattern_fp = Chem.rdmolops.PatternFingerprint(mol)
        pattern_fp_array = np.array(pattern_fp)
        return pattern_fp_array

class wrapper_MACCSFPGen(object):
    def __call__(self, mol):
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_fp_array = np.array(maccs_fp)
        return maccs_fp_array


wMolToSmiles = wrapper_MolToSmiles()
wRDKitFPGen = wrapper_RDKitFPGen()
wAtomPairFPGen = wrapper_AtomPairFPGen()
wTorsionFPGen = wrapper_TorsionFPGen()
wMorganFPGen = wrapper_MorganFPGen()
wAvalonFPGen = wrapper_AvalonFPGen()
wLayeredFPGen = wrapper_LayeredFPGen()
wPatternFPGen = wrapper_PatternFPGen()
wMACCSFPGen = wrapper_MACCSFPGen()

class generate_fingerprints(object):
    def __init__(self, smile_list, num_worker=5):
        self.smile_list = smile_list
        self.num_worker = num_worker
        self.mol_list = Parallel(n_jobs=self.num_worker)(delayed(wMolToSmiles)(smile) for smile in self.smile_list)
    
    def gen_rdkit_fp(self):
        rdkit_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wRDKitFPGen)(mol) for mol in self.mol_list)
        self.rdkit_fp_arrays = np.array(rdkit_fp_list)

    def gen_atompair_fp(self):
        atompair_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wAtomPairFPGen)(mol) for mol in self.mol_list)
        self.atompair_fp_arrays = np.array(atompair_fp_list)

    def gen_torsion_fp(self):
        torsion_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wTorsionFPGen)(mol) for mol in self.mol_list)
        self.torsion_fp_arrays = np.array(torsion_fp_list)

    def gen_morgan_fp(self):
        morgan_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wMorganFPGen)(mol) for mol in self.mol_list)
        self.morgan_fp_arrays = np.array(morgan_fp_list)
    
    def gen_avalon_fp(self):
        avalon_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wAvalonFPGen)(mol) for mol in self.mol_list)
        self.avalon_fp_arrays = np.array(avalon_fp_list)

    def gen_layered_fp(self):
        layered_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wLayeredFPGen)(mol) for mol in self.mol_list)
        self.layered_fp_arrays = np.array(layered_fp_list)
    
    def gen_pattern_fp(self):
        pattern_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wPatternFPGen)(mol) for mol in self.mol_list)
        self.pattern_fp_arrays = np.array(pattern_fp_list)

    def gen_maccs_fp(self):
        maccs_fp_list = Parallel(n_jobs=self.num_worker)(delayed(wMACCSFPGen)(mol) for mol in self.mol_list)
        self.maccs_fp_arrays = np.array(maccs_fp_list)

class wrapper_extract_murcko_scaffold(object):
    def __init__(self, method_name):
        self.method_name = method_name

    def __call__(self, smile):
        mol = Chem.MolFromSmiles(smile)
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        generic_scaffold_mol = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
        scaffold = Chem.MolToSmiles(scaffold_mol)
        generic_scaffold = Chem.MolToSmiles(generic_scaffold_mol)
        return scaffold, generic_scaffold
