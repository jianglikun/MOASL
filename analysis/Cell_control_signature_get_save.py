# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-Cell_control_signature_get_save.py
@time:2022/9/17 下午3:35
"""

import os,sys
import pandas as pd
import numpy as np
from cmapPy.pandasGEXpress.parse import parse

dir_ = '/mnt/data1/jlk/Cmap_database/CLUE/'
level3_signature_file = dir_ + 'level3_beta_ctl_n188708x12328.gctx'
level4_signature_file = dir_ + 'level4_beta_ctl_n188708x12328.gctx'
level5_signature_file = dir_ + 'level5_beta_ctl_n58022x12328.gctx'
level5_info_file = dir_ +'siginfo_beta.txt'
level5_info_df = pd.read_csv(level5_info_file,sep='\t')
level3_info_file = dir_ + 'instinfo_beta.txt'
level3_info_df = pd.read_csv(level3_info_file,sep='\t')


level3_save_dir = '{}/level3_cell_control'.format(dir_)
level3_mfc_save_dir = '{}/level3_cell_mfc_control'.format(dir_)
# def save_cell_signature_level3(signature_file,info_df,save_dir):
#     info_ctl_df = info_df[info_df['pert_type'] == 'ctl_vehicle']
#     cell_types = info_ctl_df['cell_iname'].value_counts()
#     for cell_type in cell_types.index:
#         cell_df = info_ctl_df[info_ctl_df['cell_iname']==cell_type]
#         print(cell_type,cell_df.shape)
#         df = parse(signature_file, cid=cell_df['sample_id'])
#         save_file = '{}/{}_{}.csv'.format(save_dir, cell_type, cell_df.shape[0])
#         df.data_df.to_csv(save_file)
# save_cell_signature_level3(signature_file=level3_signature_file,
#                      info_df=level3_info_df,
#                      save_dir=level3_save_dir)

def save_cell_signature_level3(signature_file,info_df,save_dir):
    info_ctl_df = info_df[info_df['pert_type'] == 'ctl_vehicle']
    cell_types = info_ctl_df['cell_mfc_name'].value_counts()
    for cell_type in cell_types.index:
        cell_df = info_ctl_df[info_ctl_df['cell_mfc_name'] == cell_type]
        save_file = '{}/{}_{}.csv'.format(save_dir, cell_type, cell_df.shape[0])
        if os.path.exists(save_file):
            pass
        else:
            print(cell_type,cell_df.shape)
            df = parse(signature_file, cid=cell_df['sample_id'])
            df.data_df.to_csv(save_file)


level4_save_dir = '{}/level4_cell_control'.format(dir_)
def save_cell_signature_level4(signature_file,info_df,save_dir):
    info_ctl_df = info_df[info_df['pert_type'] == 'ctl_vehicle']
    cell_types = info_ctl_df['cell_iname'].value_counts()
    for cell_type in cell_types.index:
        cell_df = info_ctl_df[info_ctl_df['cell_iname']==cell_type]
        sig_list = []
        for i in cell_df['distil_ids']:
            sig_list += i.split('|')
        sig_list = list(set(sig_list))
        try:
            df = parse(signature_file, cid=sig_list)
            print(cell_type,len(sig_list),cell_df.shape,df.data_df.shape)
            save_file = '{}/{}_{}.csv'.format(save_dir, cell_type, len(sig_list))
            df.data_df.to_csv(save_file)
        except:
            print(cell_type)



level5_save_dir = '{}/level5_cell_control'.format(dir_)
def save_cell_signature_level5(signature_file,info_df,save_dir):
    info_ctl_df = info_df[info_df['pert_type'] == 'ctl_vehicle']
    cell_types = info_ctl_df['cell_iname'].value_counts()
    for cell_type in cell_types.index:
        cell_df = info_ctl_df[info_ctl_df['cell_iname']==cell_type]
        print(cell_type,cell_df.shape)
        df = parse(signature_file, cid=cell_df['sig_id'])
        save_file = '{}/{}_{}.csv'.format(save_dir, cell_type, cell_df.shape[0])
        df.data_df.to_csv(save_file)


if __name__ == '__main__':
    save_cell_signature_level3(signature_file=level3_signature_file,
                               info_df=level3_info_df,
                               save_dir=level3_save_dir)
    save_cell_signature_level4(signature_file=level4_signature_file,
                               info_df=level5_info_df,
                               save_dir=level4_save_dir)
    save_cell_signature_level5(signature_file=level5_signature_file,
                               info_df=level5_info_df,
                               save_dir=level5_save_dir)