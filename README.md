## MOASL
MOASL: Prediction of the mechanism of action from transcriptional signature based on similarity learning!


##  Dependencies package:
```
* sklearn
* numpy
* pandas 
* scipy
* pytorch-metric-learning
* h5py
* faiss-gpu
```

#### Install from github   

    git clone https://github.com/jianglikun/MOASL.git

## Usage 
We present MOASL (MOAs prediction via Similarity Learning), which automatically learns similarity embedding among signatures of shared MOAs through contrastive approach. We evaluated the accuracy of signature matching using MOASL on different transcriptional activities score (TAS) datasets as well as diverse cell lines. MOASL offers substantial performance improvements over several statistical and machine learning methods. In addition, we demonstrate the rationale of the model by visualizing the signature annotation procedure, where the query signatures can easily predict the MOAs labels by calculating their similarity to the reference embedding. Finally, we applied MOASL to glucocorticoid receptor (GR) agonist and 8 compounds in top-10 are correctly defined as GR agonist.

![image](https://github.com/jianglikun/MOASL/blob/main/Figure1.png)

#### **(i)** prepare data
Download the L1000 dataset form [CLUE](https://clue.io/data/CMap2020#LINCS2020);

Use [cmapPy](https://github.com/cmap/cmapPy) process the L1000 data before you gernerating the dataset by youself;

Or you can directly use my processed data, download from https://pan.baidu.com/s/1syd05gFX7x4_STb3eVi13w (verify code 2023);

```
# prepare single signature data
python data/prepare_data_level5.py

# prepare TAS-high data
python data/prepare_data_tas_high.py

# prepare TAS-all data and signle cell data
python data/prepare_data_tas_all.py
```

#### **(ii)** train and test
```
python src/train.py
```

#### **(iii)** compare methods 
```
# all compare method script in ./compare

Benchmark.py

CMapGSEA.py

CMapKS.py

Drsim.py

Euclidean.py

RF.py

Random.py

SVM.py

Spearman.py

jaccard.py
```
##### **(iiii)** downstream analysis
* ##### Figure 2
```
# change "experiment_result.xlsx" as suuplymentary_file.xlsx, table=TableS2
plot_acc_tas.ipynb

```
* ##### Figure 3
```
analysis_drug_demention.ipynb
```
* ##### Figure 4
```
# change "experiment_result.xlsx" as suuplymentary_file.xlsx, table=TableS3
plot_acc_low.ipynb
```
* ##### Figure 5
```
cell_display.ipynb
```

## Citation
MOASL: Prediction of the mechanism of action from transcriptional signature based on similarity learning

submit in 2023


## concat
Likun Jiang jianglikun@stu.xmu.edu.cn

Xiangrong Liu xrliu@xmu.edu.cn

Xiamen University,Xiamen,China

