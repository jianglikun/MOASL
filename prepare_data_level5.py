# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-prepare_data_level4.py
@time:2022/12/21 下午8:01
"""

import os,sys
import pandas as pd
import numpy as np
import random
from cmapPy.pandasGEXpress.parse import parse

class Prepare_data():
    def __init__(self):
        # get all single signature and save
        dir_ = '/mnt/data1/jlk/Cmap_database/CLUE/'
        self.level5_signature_file = dir_ + 'level5_beta_trt_cp_n720216x12328.gctx'
        self.level5_info_file = dir_ + 'siginfo_beta.txt'
        self.compounds_info_file = dir_ + 'compoundinfo_beta.txt'
        self.gene_info_file = dir_ + 'geneinfo_beta.txt'
        self.landmark_gene = self.get_landmark()
        self.compounds_info_df = self.get_compounds()
        self.level5_info_df = self.get_level5_info()
        self.save_dir = dir_ + '/level5_single'
        self.train_test_dir = dir_ + '/level5_train_test'

    def get_landmark(self):
        gene_df = pd.read_csv(self.gene_info_file, sep='\t')
        print('###',gene_df['feature_space'].value_counts())
        landmark_gene_df = gene_df[gene_df['feature_space'] == 'landmark']
        landmark_gene_id = landmark_gene_df['gene_id']
        landmark_gene_id = list(landmark_gene_id)
        landmark_gene_id = ['{}'.format(i) for i in landmark_gene_id]
        print('### {} landmark gene record!'.format(len(landmark_gene_id)))
        return landmark_gene_id

    def get_compounds(self):
        cp_info_df = pd.read_csv(self.compounds_info_file,sep='\t')
        print(('### here {} drugs record in CLUE databse'.format(cp_info_df.shape[0])))
        cp_info_df = cp_info_df[['pert_id', 'canonical_smiles']]
        cp_info_df = cp_info_df.drop_duplicates()
        cp_info_df = cp_info_df.dropna()
        print('### there are {} compounds with smiles'.format(cp_info_df.shape[0]))
        return cp_info_df

    def get_level5_info(self):
        # signature with pert_id and simles
        level5_info_df = pd.read_csv(self.level5_info_file, sep='\t')
        level5_info_df = level5_info_df[level5_info_df['pert_type'] == 'trt_cp']
        print('### There are {} compounds signatures in level5'.format(level5_info_df.shape[0]))
        level5_info_df = level5_info_df[level5_info_df['pert_id'].isin(self.compounds_info_df['pert_id'])]
        print('### There are {} compounds signatures with simles in level5'.format(level5_info_df.shape[0]))
        return level5_info_df

    def stat_split_single(self):
        print(self.level5_info_df)
        drug_number = self.level5_info_df['pert_id'].value_counts()

        # more than 10 times,drug
        drug_number_10 = drug_number[drug_number >= 10]
        # print(drug_number_10)
        # more than 20 times,drug
        drug_number_20 = drug_number[drug_number >= 20]
        # print(drug_number_20)
        # more than 100 times,drug
        drug_number_100 = drug_number[drug_number >= 100]
        # print(drug_number_100)

        # train test split in 10 times drugs
        train_list = []
        test_list = []
        for drug in drug_number_10.index:
            sub_df = self.level5_info_df[self.level5_info_df['pert_id']==drug]
            signatures = sub_df['sig_id'].tolist()
            signatures_number = len(signatures)
            train_number = int(signatures_number*0.8)
            train_signature = random.sample(signatures,train_number)
            for i in train_signature:
                signatures.remove(i)
            test_signature = signatures
            # print(signatures_number,len(train_signture),len(test_signature))
            train_list += train_signature
            test_list += test_signature
        print('Train and Test number:',len(train_list),len(test_list))
        train_test_dict = {'train': train_list,'test':test_list}
        save_file = '{}/train_{}_test_{}_single.npz'.format(self.train_test_dir,len(train_list),len(test_list))
        np.savez(save_file,**train_test_dict)

    def stat_spit_cell(self):
        cell_list = set(self.level5_info_df['cell_iname'].tolist())
        cell_list = list(cell_list)
        print(cell_list)
        print('### here we have {} cell number'.format(len(cell_list)))
        save_cells = []
        save_drugs = []
        for cell in cell_list:
            cell_df = self.level5_info_df[self.level5_info_df['cell_iname'] == cell]
            cell_drug_df = cell_df['pert_id'].value_counts()
            cell_drug_df = cell_drug_df[cell_drug_df>10]
            cell_drug_number = cell_drug_df.shape[0]
            if cell_drug_number > 0:
                save_cells.append(cell)
                save_drugs += list(cell_drug_df.index)
        print('### {} cell line have 10 times durg!'.format(len(save_cells)))
        print('### {} drugs in cell line before duplication!'.format(len(save_drugs)))
        print('### {} drugs in cell line after drop duplication!'.format(len(set(save_drugs))))

        # train test split in 10 times drugs
        train_list = []
        test_list = []
        for cell in cell_list:
            cell_df = self.level5_info_df[self.level5_info_df['cell_iname'] == cell]
            cell_drug_df = cell_df['pert_id'].value_counts()
            cell_drug_df = cell_drug_df[cell_drug_df>10]
            for drug in cell_drug_df.index:
                drug_df = cell_df[cell_df['pert_id'] == drug]
                signatures = drug_df['sig_id'].tolist()
                signatures_number = len(signatures)
                train_number = int(signatures_number * 0.8)
                train_signature = random.sample(signatures, train_number)
                for i in train_signature:
                    signatures.remove(i)
                test_signature = signatures
                # print(signatures_number,len(train_signture),len(test_signature))
                train_list += train_signature
                test_list += test_signature
        print('Train and Test number:', len(train_list), len(test_list))
        train_test_dict = {'train': train_list, 'test': test_list}
        save_file = '{}/train_{}_test_{}_cell_{}.npz'.format(self.train_test_dir, len(train_list), len(test_list),len(save_cells))
        np.savez(save_file, **train_test_dict)


    def save_single(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # accroding sig_id get smile
        compoundsID_2_smile = dict(zip(self.compounds_info_df['pert_id'], self.compounds_info_df['canonical_smiles']))

        # all signature
        single_sig_df = parse(self.level5_signature_file)
        print(single_sig_df)
        print(single_sig_df.data_df.shape)
        print(single_sig_df.data_df)
        save_index_dict = {'index_12328':single_sig_df.data_df.index,'index_978':self.landmark_gene}
        save_index_file = '{}/index.npz'.format(self.save_dir)
        np.savez(save_index_file,**save_index_dict)

        # save every single signature
        n = 0
        for sig_id in self.level5_info_df['sig_id']:
            info = self.level5_info_df[self.level5_info_df['sig_id']==sig_id]
            if info.shape[0] >1:
                print(sig_id,info)
            pert_id = info['pert_id'].tolist()[0]
            pert_time = info['pert_time'].tolist()[0]
            pert_idose = info['pert_idose'].tolist()[0]
            cell_mfc_name = info['cell_mfc_name'].tolist()[0]
            smile = compoundsID_2_smile[pert_id]

            signature_12328 = single_sig_df.data_df[sig_id]
            signature_978 = signature_12328[self.landmark_gene]
            save_file = '{}/{}.npz'.format(self.save_dir,sig_id)
            save_dict = {'pert_id':pert_id,
                         'pert_time':pert_time,
                         'pert_idose':pert_idose,
                         'cell_name':cell_mfc_name,
                         'smile':smile,
                         'signature_12328':signature_12328,
                         'signature_978':signature_978,
                         }
            np.savez(save_file,**save_dict)
            n +=1
            if n%10000 == 0:
                print('### Done with {} signature!'.format(n))


if __name__ == '__main__':
    obj = Prepare_data()
    obj.save_single()
    # obj.stat_split_single()
    # obj.stat_spit_cell()