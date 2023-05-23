# python3
# -*- coding:utf-8 -*-
# 

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file:PycharmProjects-PyCharm-utils.py
@time:2023/4/6 下午10:10
"""
import pandas as pd
import numpy as np
import torch
import faiss
from collections import Counter
from faiss import normalize_L2
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

def sklearn_cos_search(x, k=None):
    assert len(x.shape) == 2, "仅支持2维向量的距离计算"
    nb, d = x.shape
    ag=cosine_similarity(x)
    np.argsort(-ag, axis=1)
    k_search = k if k else nb

    return np.argsort(-ag, axis=1)[:, :k_search]

def calCosine(Xtr, Xte):
    dat_cor = pd.DataFrame(cosine_similarity(Xte,Xtr))
    dat_cor.columns = Xtr.index
    dat_cor.index = Xte.index
    return dat_cor

def query_moa_cosine(query_embedding,ref_embedding,query_labels,ref_labels):
    distance_metr = cosine_similarity(query_embedding.float().cpu(),ref_embedding.float().cpu())
    # distance_metr = cosine_similarity(ref_embedding.float().cpu(),query_embedding.float().cpu())

    find_indices = np.argsort(-distance_metr, axis=1)[:, :1]
    # print(distance_metr,distance_metr.shape,find_indices.shape,find_indices)
    a = distance_metr[range(distance_metr.shape[0]),find_indices.reshape(-1)]
    # print(a,a.shape)
    find_labels = ref_labels[find_indices]
    find_labels = find_labels[:, :1]
    # print(find_labels,find_labels.shape)
    query_labels = query_labels[:, None]
    sample_num = query_embedding.shape[0]
    same_num = torch.eq(query_labels, find_labels).sum().cpu().tolist()
    percent = round(same_num/sample_num,4)*100
    print('Test {} signatures;{} correct;{} %'.format(sample_num,same_num,percent))
    return sample_num,same_num

def query_moa_function(query_embedding,ref_embedding,query_labels,ref_labels):
    k = 1
    dim = ref_embedding.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(ref_embedding.float().cpu())
    distances, indices = index.search(query_embedding.float().cpu(),k)
    # print(distances)

    find_labels = ref_labels[indices]
    find_labels = find_labels[:, :k]
    query_labels = query_labels[:, None]

    sample_num = query_embedding.shape[0]
    same_num = torch.eq(query_labels, find_labels).sum().cpu().tolist()
    percent = round(same_num/sample_num,4)*100
    print('Test {} signatures;{} correct;{} %'.format(sample_num,same_num,percent))
    return sample_num,same_num

def query_moa_high(query_embedding,ref_embedding,query_labels,ref_labels,label_dict):
    k = 1
    dim = ref_embedding.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(ref_embedding.float().cpu())
    distances, indices = index.search(query_embedding.float().cpu(),k)
    find_labels = ref_labels[indices]
    find_labels = find_labels[:, :k]
    query_labels = query_labels[:, None]

    sample_num = query_embedding.shape[0]
    same_num = torch.eq(query_labels, find_labels).sum().cpu().tolist()
    percent = round(same_num/sample_num,4)*100
    print('Test {} signatures;{} correct;{} %'.format(sample_num,same_num,percent))

    query_labels = query_labels.float().cpu()
    find_labels = find_labels.float().cpu()
    test_moa_uniq = np.unique(query_labels)
    save_df = pd.DataFrame()
    moa_li = []
    moa_num_li = []
    same_num_li = []
    per_li = []
    for moa in test_moa_uniq:
        moa_name = label_dict[moa]
        moa_idx,_ = np.where(query_labels==moa)
        moa_num = moa_idx.shape[0]
        moa_label = query_labels[moa_idx]
        moa_find = find_labels[moa_idx]
        # print(moa_name,moa,moa_num,moa_idx,moa_label,moa_find)
        same_num = torch.eq(moa_label, moa_find).sum().cpu().tolist()
        moa_percent = round(same_num/moa_num, 4) * 100
        moa_li.append(moa_name);moa_num_li.append(moa_num);same_num_li.append(same_num);per_li.append(moa_percent)
        if moa_percent >= 40:
            print('MOA #{}# {} signatures;{} correct;{} %'.format(moa_name,moa_num, same_num, moa_percent))
    save_df['moa'] = moa_li
    save_df['number'] = moa_num_li
    save_df['find'] = same_num_li
    save_df['per'] = per_li

    return save_df

def query_drug_cosine_bk(query_embedding,ref_embedding,query_labels,ref_labels,
                      drug_label_dict,drug2moa_dict,moa_label_dict):
    query_drugs = np.array([[drug_label_dict[i]] for i in query_labels.cpu().tolist()])
    ref_drugs = np.array([[drug_label_dict[i]] for i in ref_labels.cpu().tolist()])

    ref_moas = np.array([moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in ref_labels.cpu().tolist()])
    query_moas = np.array([moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in query_labels.cpu().tolist()])

    moa_rev_label_dict = dict([val, key] for key, val in moa_label_dict.items())

    distance_metr = cosine_similarity(query_embedding.float().cpu(), ref_embedding.float().cpu())
    # distance_metr = cosine_similarity(ref_embedding.float().cpu(),query_embedding.float().cpu())

    find_indices = np.argsort(-distance_metr, axis=1)[:, :1]
    # print(distance_metr, distance_metr.shape, find_indices.shape, find_indices)
    query_distance = distance_metr[range(distance_metr.shape[0]), find_indices.reshape(-1)]
    # print(a, a.shape)
    find_labels = ref_labels[find_indices]
    find_labels = find_labels[:, :1]
    query_labels = query_labels[:, None]

    find_moas = ref_moas[find_indices]
    find_drugs = ref_drugs[find_indices]
    find_moas = find_moas[:, :1]
    find_drugs = ref_drugs[:,:1]
    query_moas = query_moas[:, None]

    sample_num_all = query_embedding.shape[0]
    same_num_all = np.equal(query_moas, find_moas).sum().tolist()
    percent = round(same_num_all/sample_num_all,4)*100
    print('### All Test {} signatures;{} correct;{} %'.format(sample_num_all,same_num_all,percent))

    test_moa_uniq = np.unique(query_moas)
    test_drug_uniq = np.unique(query_drugs)
    print('Test {} drug'.format(test_drug_uniq.shape[0]))
    print(test_moa_uniq)
    drug_list = []
    drug_sig_num_list = []
    drug_find_num_list = []
    drug_label_list = []
    drug_find_label_list = []
    drug_find_dis_list = []
    drug_rev_label_list = []
    drug_rev_dis_list = []
    save_df = pd.DataFrame()
    for moa in test_moa_uniq:
        moa_idx, _ = np.where(query_moas == moa)
        moa_num = moa_idx.shape[0]
        moa_label = query_moas[moa_idx]
        moa_find = find_moas[moa_idx]
        # print(moa_name,moa,moa_num,moa_idx,moa_label,moa_find)
        find_moa_num = np.equal(moa_label, moa_find).sum().tolist()
        moa_percent = round(find_moa_num/ moa_num, 4) * 100
        print('### MOA {} signatures;{} correct;{} %'.format(moa, find_moa_num, moa_percent))

        for drug in test_drug_uniq:
            moa_str = str(drug2moa_dict[drug])
            if moa_label_dict[moa_str] == moa:
                drug_idx,_ = np.where(query_drugs==drug)
                drug_num = drug_idx.shape[0]
                drug_distance = query_distance[drug_idx]
                moa_label = query_moas[drug_idx]
                moa_find = find_moas[drug_idx]

                print(moa)

                find_idx = np.equal(moa_label,moa_find).reshape(-1)
                find_dis = np.mean(drug_distance[find_idx])
                rev_dis = np.mean(drug_distance[~find_idx])

                same_num = np.equal(moa_label, moa_find).sum().tolist()
                moa_percent = round(same_num / drug_num, 4) * 100

                m_coun = Counter(list(moa_find.reshape(-1)))
                if same_num == drug_num or same_num == 0:
                    most_moa = m_coun.most_common(1)[0][0]
                    most_moa = moa_rev_label_dict[most_moa]
                    rev_moa = '-'
                else:
                    most_moa = m_coun.most_common(1)[0][0]
                    most_moa = moa_rev_label_dict[most_moa]
                    rev_moa = m_coun.most_common(2)[1][0]
                    rev_moa = moa_rev_label_dict[rev_moa]
                drug_list.append(drug)
                drug_sig_num_list.append(drug_num)
                drug_find_num_list.append(same_num)
                drug_label_list.append(moa_rev_label_dict[moa_label[0][0]])
                drug_find_label_list.append(most_moa)
                drug_find_dis_list.append(find_dis)
                drug_rev_label_list.append(rev_moa)
                drug_rev_dis_list.append(rev_dis)
                # print('{} drug # {} # has {} signatures;{} correct;{} %; distance {}'.format(
                #     moa_str,drug,drug_num, same_num, moa_percent,np.mean(drug_distance)))
    save_df['drug'] = drug_list
    save_df['sig_num'] = drug_sig_num_list
    save_df['find_num'] = drug_find_num_list
    save_df['find_distance'] = drug_find_dis_list
    save_df['true_moa'] = drug_label_list
    save_df['final_moa'] = drug_find_label_list
    save_df['reverse_moa'] =  drug_rev_label_list
    save_df['reverse_distance'] = drug_rev_dis_list
    # train_index = [str(drug2moa_dict[drug_label_dict[i]])+'_'+drug_label_dict[i] for i in ref_labels.cpu().tolist()]
    # save_train_df = pd.DataFrame(ref_embedding.cpu(),index=train_index)
    #
    # test_index = [str(drug2moa_dict[drug_label_dict[i]])+'_'+drug_label_dict[i] for i in query_labels.cpu().tolist()]
    # save_test_df = pd.DataFrame(query_embedding.cpu(),index=test_index)


    return sample_num_all,same_num_all,save_df

def query_drug_cosine(query_embedding,ref_embedding,query_labels,ref_labels,
                      drug_label_dict,drug2moa_dict,moa_label_dict,moa_aim,cp_info_df):
    query_drugs = np.array([[drug_label_dict[i]] for i in query_labels.cpu().tolist()])
    ref_drugs = np.array([[drug_label_dict[i]] for i in ref_labels.cpu().tolist()])

    ref_moas = np.array([moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in ref_labels.cpu().tolist()])
    query_moas = np.array([moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in query_labels.cpu().tolist()])

    moa_rev_label_dict = dict([val, key] for key, val in moa_label_dict.items())

    distance_metr = cosine_similarity(query_embedding.float().cpu(), ref_embedding.float().cpu())

    find_indices = np.argsort(-distance_metr, axis=1)[:, :1]
    query_distance = distance_metr[range(distance_metr.shape[0]), find_indices.reshape(-1)]

    find_moas = ref_moas[find_indices]
    find_moas = find_moas[:, :1]
    query_moas = query_moas[:, None]

    sample_num_all = query_embedding.shape[0]
    same_num_all = np.equal(query_moas, find_moas).sum().tolist()
    percent = round(same_num_all/sample_num_all,4)*100
    print('### All Test {} signatures;{} correct;{} %'.format(sample_num_all,same_num_all,percent))

    test_moa_uniq = np.unique(query_moas)
    test_drug_uniq = np.unique(query_drugs)
    print('### Test {} drug'.format(test_drug_uniq.shape[0]))

    for moa in test_moa_uniq:
        moa_idx, _ = np.where(query_moas == moa)
        moa_num = moa_idx.shape[0]
        moa_label = query_moas[moa_idx]
        moa_find = find_moas[moa_idx]
        # print(moa_name,moa,moa_num,moa_idx,moa_label,moa_find)
        find_moa_num = np.equal(moa_label, moa_find).sum().tolist()
        moa_percent = round(find_moa_num/ moa_num, 4) * 100
        print('### MOA {} has {} signatures {} % correct;'.format(moa_rev_label_dict[moa], moa_num, moa_percent))

    drug_list = []
    drug_sig_num_list = []
    drug_pos_num_list = []
    drug_pos_per_list = []
    drug_pos_dis_list = []
    drug_neg_num_list = []
    drug_neg_dis_list = []
    drug_label_list = []
    drug_find_label_list = []
    save_df = pd.DataFrame()
    for drug in test_drug_uniq:
        moa_str = str(drug2moa_dict[drug])
        drug_idx, _ = np.where(query_drugs == drug)
        drug_num = drug_idx.shape[0]
        drug_distance = query_distance[drug_idx]
        moa_label = query_moas[drug_idx]
        moa_find = find_moas[drug_idx]

        neg_idx = np.equal(moa_label_dict['negative'],moa_find).reshape(-1)
        neg_dis = np.mean(drug_distance[neg_idx])
        neg_num = np.equal(moa_find,moa_label_dict['negative']).sum().tolist()
        pos_idx = np.equal(moa_label_dict[moa_aim], moa_find).reshape(-1)
        pos_dis = np.mean(drug_distance[pos_idx])
        pos_num = np.equal(moa_label_dict[moa_aim], moa_find).sum().tolist()
        pos_per = round(pos_num/drug_num,2)
        # print(drug_num,neg_num,neg_dis,pos_num,pos_dis)

        if pos_num >= neg_num and pos_dis > neg_dis:
            label = moa_aim
        else:
            label = 'negative'

        drug_list.append(drug)
        drug_sig_num_list.append(drug_num)
        drug_pos_num_list.append(pos_num)
        drug_pos_per_list.append(pos_per)
        drug_neg_num_list.append(neg_num)
        drug_pos_dis_list.append(pos_dis)
        drug_neg_dis_list.append(neg_dis)
        drug_label_list.append(moa_str)
        drug_find_label_list.append(label)

    save_df['drug'] = drug_list
    save_df['sig_num'] = drug_sig_num_list
    save_df['pos_num'] = drug_pos_num_list
    save_df['neg_num'] = drug_neg_num_list
    save_df['pos_per'] = drug_pos_per_list
    save_df['pos_distance'] = drug_pos_dis_list
    save_df['neg_distance'] = drug_neg_dis_list
    save_df['true_moa'] = drug_label_list
    save_df['final_moa'] = drug_find_label_list

    # train_index = [str(drug2moa_dict[drug_label_dict[i]])+'_'+drug_label_dict[i] for i in ref_labels.cpu().tolist()]
    # save_train_df = pd.DataFrame(ref_embedding.cpu(),index=train_index)
    #
    # test_index = [str(drug2moa_dict[drug_label_dict[i]])+'_'+drug_label_dict[i] for i in query_labels.cpu().tolist()]
    # save_test_df = pd.DataFrame(query_embedding.cpu(),index=test_index)


    return sample_num_all,same_num_all,save_df

def query_drug_high(query_embedding,ref_embedding,query_labels,ref_labels,drug_label_dict,drug2moa_dict,moa_label_dict):
    query_drugs = np.array([[drug_label_dict[i]] for i in query_labels.cpu().tolist()])
    ref_drugs = np.array([[drug_label_dict[i]] for i in ref_labels.cpu().tolist()])

    ref_moas = np.array([moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in ref_labels.cpu().tolist()])
    query_moas = np.array([moa_label_dict[str(drug2moa_dict[drug_label_dict[i]])] for i in query_labels.cpu().tolist()])

    k = 1
    dim = ref_embedding.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(ref_embedding.float().cpu())
    distances, indices = index.search(query_embedding.float().cpu(),k)
    print(indices)

    find_moas = ref_moas[indices]
    find_drugs = ref_drugs[indices]
    find_moas = find_moas[:, :k]
    find_drugs = ref_drugs[:,:k]
    query_moas = query_moas[:, None]

    sample_num = query_embedding.shape[0]
    same_num = np.equal(query_moas, find_moas).sum().tolist()
    percent = round(same_num/sample_num,4)*100
    print('Test {} signatures;{} correct;{} %'.format(sample_num,same_num,percent))

    test_moa_uniq = np.unique(query_moas)
    test_drug_uniq = np.unique(query_drugs)
    for moa in test_moa_uniq:
        moa_idx, _ = np.where(query_moas == moa)

        moa_num = moa_idx.shape[0]
        moa_label = query_moas[moa_idx]
        moa_find = find_moas[moa_idx]
        # print(moa_name,moa,moa_num,moa_idx,moa_label,moa_find)
        find_moa_num = np.equal(moa_label, moa_find).sum().tolist()
        moa_percent = round(find_moa_num/ moa_num, 4) * 100
        if moa_percent >= 40:
            print('### MOA {} signatures;{} correct;{} %'.format(moa_num, find_moa_num, moa_percent))
            for drug in test_drug_uniq:
                moa_str = str(drug2moa_dict[drug])
                if moa_label_dict[moa_str] == moa:
                    drug_idx,_ = np.where(query_drugs==drug)
                    drug_num = drug_idx.shape[0]
                    moa_label = query_moas[drug_idx]
                    moa_find = find_moas[drug_idx]
                    drug_find = find_drugs[drug_idx]

                    # print(moa_name,moa,moa_num,moa_idx,moa_label,moa_find)
                    same_num = np.equal(moa_label, moa_find).sum().tolist()
                    moa_percent = round(same_num / drug_num, 4) * 100
                    if moa_percent >= 70 :
                        # print(drug, drug2moa_dict[drug], moa)
                        # print(moa_find,drug_find)
                        print('{} drug # {} # has {} signatures;{} correct;{} %'.format(moa_str,drug,drug_num, same_num, moa_percent))

    train_index = [str(drug2moa_dict[drug_label_dict[i]])+'_'+drug_label_dict[i] for i in ref_labels.cpu().tolist()]
    save_train_df = pd.DataFrame(ref_embedding.cpu(),index=train_index)

    test_index = [str(drug2moa_dict[drug_label_dict[i]])+'_'+drug_label_dict[i] for i in query_labels.cpu().tolist()]
    save_test_df = pd.DataFrame(query_embedding.cpu(),index=test_index)


    return sample_num,same_num,save_train_df,save_test_df