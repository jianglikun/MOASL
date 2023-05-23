import argparse
parser = argparse.ArgumentParser(description='It is used for drug annotation',formatter_class=argparse.RawDescriptionHelpFormatter,add_help=True)

Req = parser.add_argument_group('Required')
Req.add_argument('-ref', metavar='\b', action='store', help='reference used for training', required=False )
Req.add_argument('-query', metavar='\b', action='store', help='query used for assignment', required=False )

Opt = parser.add_argument_group('Optional')
Opt.add_argument('-pvalue', metavar='\b', action='store', default=0.01, type = float, help='pvalue, default: 0.01')
Opt.add_argument('-variance', metavar='\b', action='store', type = float, default=0.98, help='variance to keep, default: 0.98')
Opt.add_argument('-dimension', metavar='\b', action='store', type = int, default=50, help='dimension of LDA, default: 50')
Opt.add_argument('-output', metavar='\b', action='store', type = str, default='DrSim.tsv', help='outfile prefix, default: DrSim.tsv')
Opt.add_argument('-num', metavar='\b', action='store', type = int, default=10, help='number of results to keep, default: 10')


args = parser.parse_args()
import os, subprocess
import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from Drsim_util import calCosine, drugTOMOA, sigidTo
from itertools import product, chain
from sklearn.metrics.pairwise import cosine_similarity


class RunMultiProcess(object):
    def __init__(self, methods=''):
        self.KNN_size = 3;
        self.singleLabel = True;
        self.MOA = ''  ###  MOA, ATC
        self.landmarker = ''  ### lm
        self.methods = methods;
        self.level = 'L4';
        self.query_set = 'CCLE'
        self.ref_set = 'GDSC_ChEMBL_CTRP'  ####  GDSC  ChEMBL  CTRP
        self.GSEs = ['GSE92742']
        # self.cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20', 'VCAP', 'HCC515', 'HEPG2', 'HA1E', 'NPC']
        self.cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20', 'VCAP', 'HCC515', 'HEPG2']
        self.trTimes = ['24H', '6H']
        if self.methods:
            self.mylist = list(product(self.GSEs, self.cell_lines, self.trTimes, self.methods))
        else:
            self.mylist = list(product(self.GSEs, self.cell_lines, self.trTimes))

    def parameter(self):
        return self.mylist

        return results

Datapath = os.path.dirname(os.path.abspath(__file__))

### output ranked drug annotation result
def writeResult(dat_cor, output_file):
    drug2MOA = drugTOMOA()
    with open(output_file, 'w') as fout:
        a = ['Query'] + ['MOA_' + str(i) for i in range(1, args.num +1 )] + ['MOAScore_' + str(i) for i in range(1, args.num+1)]
        fout.write('{}\n'.format('\t'.join(a)))
        for i in dat_cor.index:
            tmp = dat_cor.loc[i, :]
            positive = tmp.sort_values(ascending=False)[:args.num].index.tolist()
            positive =  [drug2MOA.get(i, 'unKnown') for i in positive]
            values = tmp.sort_values(ascending=False)[:args.num].values.tolist()
            values = [str(round(i,4)) for i in values]
            fout.write('{}\t{}\t{}\n'.format(i, '\t'.join(positive), '\t'.join(values)))

### LDA main function
def runLDA(ref,query,sig2moa,drug2moa):
    Xtr = pd.read_hdf(ref)
    Xte = pd.read_hdf(query, key='dat')

    train_num = Xtr.shape[0]
    test_num = Xte.shape[0]
    print(Xtr.shape,Xte.shape)

    pert_iname = [sig2moa[i][0] for i in Xtr.index]
    ### use drug name as the training label

    pca = PCA(random_state=2020, n_components=args.variance)
    Xtr_pca = pca.fit_transform(Xtr)
    Xte_pca = pca.transform(Xte)
    print('### PCA shape:',Xtr_pca.shape,Xte_pca.shape)
    labelencoder = LabelEncoder()
    ytr = labelencoder.fit_transform(pert_iname)
    print('### there are {} drugs in ref!'.format(max(ytr)))
    ##############
    # dont use pca
    # Xtr_pca = Xtr
    # Xte_pca = Xte
    ##############
    if max(ytr) < args.dimension:
        dimension = max(ytr) -1
    else:
        dimension = args.dimension
    ml = LinearDiscriminantAnalysis(solver='svd', n_components=dimension)
    Xtr_pca_lda = ml.fit_transform(Xtr_pca, ytr)
    Xte_pca_lda = ml.transform(Xte_pca)
    print('### PCA LDA shape:',Xtr_pca_lda.shape,Xte_pca_lda.shape)
    Xtr_pca_lda = Xtr_pca_lda[:, ~np.isnan(Xtr_pca_lda)[0]] ## filter NA column
    Xte_pca_lda = Xte_pca_lda[:, ~np.isnan(Xte_pca_lda)[0]] ## filter NA column

    ref = pd.DataFrame(Xtr_pca_lda, index = pert_iname)

    ref = ref.groupby(pert_iname).median()
    print('### ref shape is : ',ref.shape)
    query = pd.DataFrame(data = Xte_pca_lda, index = Xte.index)
    print('### query shape is :',query.shape)

    find_num = precision_sig(ref,query,1,sig2moa,drug2moa)

    return train_num,test_num,find_num,round(find_num/test_num,4)

def precision_sig(Xtr,Xte,top,sig2moa,drug2moa):
    dat_cor = pd.DataFrame(cosine_similarity(Xte, Xtr))
    dat_cor.columns = Xtr.index # drug name
    dat_cor.index = Xte.index   # signature id

    correct_drug_num = 0
    correct_moa_num = 0
    query_num = dat_cor.shape[0]
    # print(dat_cor.shape)
    test_redefine_df = pd.DataFrame()
    sig_list = []
    drug_list = []
    moa_list = []
    pred_drug_list = []
    pred_moa_list = []

    for sig in dat_cor.index:
        tmp = dat_cor.loc[sig,:]
        find_drug = tmp.sort_values(ascending=False)[:top].index.tolist()
        find_moa = [drug2moa[i].tolist() for i in find_drug]

        # print(find_drug, find_moa)
        pred_drug_list.append(find_drug[0])
        pred_moa_list.append(find_moa[0])

        true_drug = sig2moa[sig][0]
        true_moa = sig2moa[sig][1]

        sig_list.append(sig)
        drug_list.append(true_drug)
        moa_list.append(true_moa)

        if true_drug in find_drug:
            correct_drug_num +=1

        if true_moa in find_moa:
            correct_moa_num += 1
    print(correct_drug_num,correct_moa_num,query_num,round(correct_moa_num/query_num,4))

    test_redefine_df['signature'] = sig_list
    test_redefine_df['drug'] = drug_list
    test_redefine_df['moa'] = moa_list
    test_redefine_df['pred_drug'] = pred_drug_list
    test_redefine_df['pred_moa'] = pred_moa_list

    n = 0
    t = 0
    for drug in set(drug_list):
        test_sub_df = test_redefine_df[test_redefine_df['drug']==drug]
        true_moa = drug2moa_dict[drug].tolist()
        drug_test_sig_num = test_sub_df.shape[0]
        true_recall_drug_sig_num = test_sub_df[test_sub_df['pred_moa']==true_moa].shape[0]
        if true_recall_drug_sig_num > drug_test_sig_num/2:
            n +=1

        pred_moa_count = test_sub_df['pred_moa'].value_counts().index[0]
        if pred_moa_count == true_moa:
            t +=1
    print(n,t,len(set(drug_list)),round(t/len(set(drug_list)),4))

    return correct_moa_num


if __name__ == '__main__':
    data_dir = '/home/jlk/Project/111_Cmap/MOA/data/'
    data_info_dir = '{}/01_Info_file/'.format(data_dir)
    data_all_dir = '{}/02_All_data/'.format(data_dir)
    data_cell_dir = '{}/03_Single_Cell/'.format(data_dir)

    sig2drugmoa_file = '{}/sig2drugmoa.npz'.format(data_info_dir)
    drug2moa_file = '{}/drug2moa.npz'.format(data_info_dir)
    sig2drugmoa_dict = np.load(sig2drugmoa_file)
    drug2moa_dict = np.load(drug2moa_file)

    cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'BT20',
                  'HCC515', 'HEPG2', 'HA1E', 'NPC', 'VCAP']

    # save_file = 'result/drsim.csv'
    # f = open(save_file, 'w')

    for fold in [0]:
        train_file = '{}/train_fold_{}_balance_100_simu.h5'.format(data_all_dir,fold)
        test_file = '{}/test_fold_{}_balance_100.h5'.format(data_all_dir,fold)
        ref = train_file
        query = test_file
        cell = 'ALL'
        print('#'*20,'ALL','#'*20)
        train_num,test_num,find_num,score =runLDA(ref, query, sig2drugmoa_dict, drug2moa_dict)
        # f.write('drsim,{},{},{},{},{},{}\n'.format(cell, fold, train_num, test_num, find_num, score))
