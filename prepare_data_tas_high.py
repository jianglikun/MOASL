# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-prepare_data_tas_all.py
@time:2023/4/6 下午9:54
"""

import pandas as pd
import numpy as np
import os,sys
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold


def save_label_dict(data_info_dir):
    moa_label_file = '{}/Huang_MOA_label.xlsx'.format(data_info_dir)
    compounds_info_file = '{}/compoundinfo_beta.txt'.format(data_info_dir)
    level5_sig_info_file = '{}/siginfo_beta.txt'.format(data_info_dir)

    # moa label file
    moa_df = pd.read_excel(moa_label_file, header=1)
    print('#'*50)
    print('### MOA file have {} drugs'.format(moa_df.shape[0]))
    moa_number = len(set(moa_df['MOA']))
    print('### MOA file have {} moas'.format(moa_number))
    moa_counts = moa_df['MOA'].value_counts()
    large_3drugs_moa = moa_counts[moa_counts>=3]
    print('### MOA file have {} moas large 3 drugs'.format(large_3drugs_moa.shape[0]))
    moa_df = moa_df[moa_df['MOA'].isin(large_3drugs_moa.index)]
    print('### MOA file have {} drugs'.format(moa_df.shape[0]))
    moa_number = len(set(moa_df['MOA']))
    print('### MOA file have {} moas'.format(moa_number))

    # compounds file
    cp_info_df = pd.read_csv(compounds_info_file, sep='\t')
    print('#'*50)
    print(('### here {} drugs record in CLUE databse'.format(cp_info_df.shape[0])))
    cp_info_df = cp_info_df[['pert_id', 'canonical_smiles']]
    cp_info_df = cp_info_df.drop_duplicates()
    cp_info_df = cp_info_df.dropna()
    print('### there are {} compounds with smile and NA'.format(cp_info_df.shape[0]))
    cp_info_df = cp_info_df[cp_info_df['canonical_smiles'] != 'restricted']
    print('### there are {} compounds with smiles'.format(cp_info_df.shape[0]))


    # Cmap level5 file
    level5_info_df = pd.read_csv(level5_sig_info_file, sep='\t', low_memory=False)
    level5_info_df = level5_info_df[level5_info_df['pert_type'] == 'trt_cp']
    print('#' * 50)
    print('### There are {} compounds signatures in level5'.format(level5_info_df.shape[0]))
    # level5_info_df = level5_info_df[level5_info_df['cell_iname'].isin(cell_lines)]
    # print('### There are {} compounds signatures in level5 and in 11 cell lines'.format(level5_info_df.shape[0]))
    # pert_time = [24]
    # level5_info_df = level5_info_df[level5_info_df['pert_time'].isin(pert_time)]
    # print('### There are {} compounds signatures in level5 and in 6h/24h'.format(level5_info_df.shape[0]))

    # cell_counts = level5_info_df['cell_iname'].value_counts()
    # top_10_cell = cell_counts.index[:10]
    # level5_info_df = level5_info_df[level5_info_df['cell_iname'].isin(top_10_cell)]
    # print('### There are {} compounds signatures in top 10 cells'.format(level5_info_df.shape[0]))


    level5_info_df = level5_info_df[level5_info_df['pert_id'].isin(moa_df['BRD-ID'])]
    print('### There are {} compounds signatures in level5 with moas'.format(level5_info_df.shape[0]))
    level5_info_df = level5_info_df[level5_info_df['pert_id'].isin(cp_info_df['pert_id'])]
    print('### There are {} compounds signatures with simles in level5'.format(level5_info_df.shape[0]))
    level5_info_df = level5_info_df[level5_info_df['is_hiq'] == 1]
    print('### There are {} compounds signatures high quality'.format(level5_info_df.shape[0]))
    level5_info_df = level5_info_df[level5_info_df['tas'] >= 0.2]
    print('### There are {} compounds signatures high tas'.format(level5_info_df.shape[0]))



    # save label file
    drug2moa_dict = dict(zip(list(moa_df['BRD-ID']), (moa_df['MOA'])))
    sig2drugmoa_dict = {}
    level5_info_df['moa'] = level5_info_df['pert_id'].map(drug2moa_dict)
    for id, row in level5_info_df.iterrows():
        sig2drugmoa_dict[row['sig_id']] = [row['pert_id'], row['moa']]
    save_file = '{}/sig2drugmoa_tas.npz'.format(data_info_dir)
    np.savez(save_file, **sig2drugmoa_dict)
    save_file = '{}/drug2moa_tas.npz'.format(data_info_dir)
    np.savez(save_file, **drug2moa_dict)

    return moa_df,level5_info_df,sig2drugmoa_dict,drug2moa_dict

def get_all_sigdata(sig_info_df,data_tas_dir):
    CLUE_level5_single_dir = '/mnt/data1/jlk/Cmap_database/CLUE/level5_single'
    sig_data_df = pd.DataFrame()
    for sig in sig_info_df['sig_id']:
        aim_path = '{}/{}.npz'.format(CLUE_level5_single_dir, sig)
        data = np.load(aim_path)
        sig_data = data['signature_12328']
        sig_data_df[sig] = sig_data
    index_file = '{}/index.npz'.format(CLUE_level5_single_dir)
    index_data = np.load(index_file, allow_pickle=True)
    sig_data_df.index = list(index_data['index_12328'])
    sig_data_df = sig_data_df.T
    print(sig_data_df.shape)
    save_file = '{}/all_signature_tas.h5'.format(data_tas_dir)
    sig_data_df.to_hdf(save_file,key = 'dat')

def all_data_kfold(moa_df,sig_info_df,data_tas_dir):
    # load all data
    all_data_file = '{}/all_signature_tas.h5'.format(data_tas_dir)
    all_data_df = pd.read_hdf(all_data_file)
    KFold = StratifiedKFold(n_splits=3,shuffle=True,random_state=2023)
    kfold_data = KFold.split(moa_df['BRD-ID'], moa_df['MOA'])
    print('#' * 50)
    print('### All data will split in 3 fold, test drug unseen in train data')
    for k,(train_id,test_id) in enumerate(kfold_data):
        print('### Fold {}, there are {} train drug and {} test drug'.format(k,len(train_id),len(test_id)))
        train_moa_df = moa_df.iloc[train_id]
        test_moa_df = moa_df.iloc[test_id]
        train_drugs = train_moa_df['BRD-ID']
        test_drugs = test_moa_df['BRD-ID']

        minsize_dup = 10
        sig_info_drug_counts = sig_info_df['pert_id'].value_counts()
        sig_info_drug_counts = sig_info_drug_counts[sig_info_drug_counts >= minsize_dup]
        sig_info_df = sig_info_df[sig_info_df['pert_id'].isin(sig_info_drug_counts.index)]

        train_info_df = sig_info_df[sig_info_df['pert_id'].isin(train_drugs)]
        test_info_df = sig_info_df[sig_info_df['pert_id'].isin(test_drugs)]

        train_data_df = all_data_df.loc[train_info_df['sig_id']]
        test_data_df = all_data_df.loc[test_info_df['sig_id']]

        train_save_file = '{}/Train_fold_{}.h5'.format(data_tas_dir,k)
        test_save_file = '{}/Test_fold_{}.h5'.format(data_tas_dir,k)

        train_data_df.to_hdf(train_save_file,key = 'dat')
        test_data_df.to_hdf(test_save_file,key = 'dat')
        print('### Fold {} saved, train & test drugs is:'.format(k),len(set(train_info_df['pert_id'])),len(set(test_info_df['pert_id'])))
        print('### Fold {} saved, train & test signature is:'.format(k),train_data_df.shape,test_data_df.shape)


if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/06_TAS/'.format(data_dir)

    # save data
    moa_df,level5_info_df,sig2drugmoa_dict,drug2moa_dict = save_label_dict(data_info_dir)
    # get_all_sigdata(level5_info_df,data_tas_dir)

    # make all data cross cell line, Train and Test
    all_data_kfold(moa_df,level5_info_df,data_tas_dir)
    #
    # single cell line ,Train and Test
    # cell_train_test(data_cell_dir,data_all_dir,moa_df,level5_info_df)