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


def all_balance_data_kfold(moa_df,sig_info_df,data_all_dir,sig2drugmoa_dict):
    # load all data
    all_data_file = '{}/all_signature_tas.h5'.format(data_all_dir)
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
        train_info_df['moa'] = [sig2drugmoa_dict[drug][1] for drug in train_info_df['sig_id']]

        test_info_df = sig_info_df[sig_info_df['pert_id'].isin(test_drugs)]
        test_info_df['moa'] = [sig2drugmoa_dict[drug][1] for drug in test_info_df['sig_id']]

        moa_count = train_info_df['moa'].value_counts()
        choose_moa = moa_count.index
        # train_info_df = train_info_df[train_info_df['moa'].isin(choose_moa)]
        # test_info_df = test_info_df[test_info_df['moa'].isin(choose_moa)]

        train_ori_sig = []
        train_1000_sig = []
        train_500_sig = []
        train_200_sig = []
        train_100_sig = []
        for moa in choose_moa:
            sub_train = train_info_df[train_info_df['moa']==moa]
            train_ori_sig += sub_train['sig_id'].tolist()
            sig_num = sub_train.shape[0]
            if sig_num >=1000:
                train_1000_sig += sub_train.sample(1000)['sig_id'].tolist()
            else:
                train_1000_sig += sub_train['sig_id'].tolist()
            if sig_num >=500:
                train_500_sig += sub_train.sample(500)['sig_id'].tolist()
            else:
                train_500_sig += sub_train['sig_id'].tolist()
            if sig_num >=200:
                train_200_sig += sub_train.sample(200)['sig_id'].tolist()
            else:
                train_200_sig += sub_train['sig_id'].tolist()
            if sig_num >=100:
                train_100_sig += sub_train.sample(100)['sig_id'].tolist()
            else:
                train_100_sig += sub_train['sig_id'].tolist()

        print(len(train_ori_sig),len(train_1000_sig),len(train_500_sig),len(train_100_sig))
        train_ori_df = all_data_df.loc[train_ori_sig]
        train_1000_df = all_data_df.loc[train_1000_sig]
        train_500_df = all_data_df.loc[train_500_sig]
        train_200_df = all_data_df.loc[train_200_sig]
        train_100_df = all_data_df.loc[train_100_sig]

        train_save_file = '{}/few_data/train_fold_{}_few_ori.h5'.format(data_all_dir,k)
        train_ori_df.to_hdf(train_save_file,key = 'dat')

        train_save_file = '{}/few_data/train_fold_{}_few_1000.h5'.format(data_all_dir,k)
        train_1000_df.to_hdf(train_save_file,key = 'dat')

        train_save_file = '{}/few_data/train_fold_{}_few_500.h5'.format(data_all_dir,k)
        train_500_df.to_hdf(train_save_file,key = 'dat')

        train_save_file = '{}/few_data/train_fold_{}_few_200.h5'.format(data_all_dir,k)
        train_200_df.to_hdf(train_save_file,key = 'dat')

        train_save_file = '{}/few_data/train_fold_{}_few_100.h5'.format(data_all_dir,k)
        train_100_df.to_hdf(train_save_file,key = 'dat')

        test_data_df = all_data_df.loc[test_info_df['sig_id']]
        test_save_file = '{}/few_data/test_fold_{}_few.h5'.format(data_all_dir,k)
        test_data_df.to_hdf(test_save_file,key = 'dat')

if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)
    data_tas_dir = '{}/06_TAS/'.format(data_dir)

    # save data
    moa_df,level5_info_df,sig2drugmoa_dict,drug2moa_dict = save_label_dict(data_info_dir)

    # make all data cross cell line, Train and Test
    all_balance_data_kfold(moa_df,level5_info_df,data_tas_dir,sig2drugmoa_dict)

