# DearWXpub
## wx keras version for everyone
## We show the feature selection and Cancer/Normal classification result on TCGA dataset

A Keras implementation of Wx in preprint, :   
**[Wx: a neural network-based feature selection algorithm for next-generation sequencing data Sungsoo Park, Bonggun Shin, Yoonjung Choi, Kilsoo Kang, and Keunsoo Kang]**
(https://www.biorxiv.org/content/biorxiv/early/2017/11/18/221911.full.pdf)   

 
**Differences with the paper:**   
- We use the Keras as Neural Network running framework, original paper used naive tensorflow framework
- Seleted Features can be diffent as which backend learning framework used
- Some samples inserted, so we have more samples than paper.

**Contacts**
- Your contributions to the repo are always welcome. 
Open an issue or contact me with E-mail `sspark@deargen.me`


## Usage

**Step 1.**
```
Experienment Environments

$ python 3.4
$ tensorflow gpu 1.4.0
$ keras 2.1.2
```

```
Install instructions

$ pip install tensorflow-gpu
$ pip install keras
```

**Step 2. Clone this repository to local.**
```
$ git clone https://github.com/deargen/DearWXpub.git
$ cd DearWXpub
```


**Step 3. Download a TCGA RNA-seq data**

1. Download rna-seq via TCGA-Assembler tool

  - we downloaded tool (`Module_A.R`) via  http://www.compgenome.org/TCGA-Assembler/index.php
  
2. Run `tcga_download.R`
  ( you may have to install releative packages )
  you can see the 'TCGA_DATAS' folder in current DearWXpub path.
 

**Step 4. Do the Feature selection and Get Classification Accuracy**

```
$ python wx_tcga.py
```
It will generate preprocessed TCGA data set. And Select features, Get the scores.


## Results

#### Selected Gene Markers
***Keras Wx 14***
`['EEF1A1','FN1','GAPDH','SFTPC','AHNAK','KLK3','UMOD','CTSB','COL1A1','GPX3','GNAS','ATP1A1','SFTPB','ACTB']`

***Peng 14***
`['KIF4A','NUSAP1','HJURP','NEK2','FANCI','DTL','UHRF1','FEN1','IQGAP3','KIF20A','TRIM59','CENPL','C16ORF59','UBE2C']`

***edgeR 14***
`['LCN1','UMOD','AQP2','PATE4','SLC12A1','OTOP2','ACTN3','KRT36','ATP2A1','PRH2','AGER','PYGM','PRR4','ESRRB']`

#### Cancer Classifiation Accuracy each type
TCGA data( Downloaded at Dec. 26th. 2017 )

|       |         | Wx 14  |       | Peng 14 |       | EdgeR 14 |        |
|:-----:|---------|--------|-------|---------|-------|----------|--------|
| TYPE  | SAMPLES | Hit    | Acc(%)| Hit     | Acc(%)| Hit      | Acc(%) |
| **TOTAL** | **3119** | **3015** | **96.67** | **2961** | **94.93** | **2957** | **94.81** |
| BLCA  | 214     | 205    | 95.79 | 208     | 97.20 | 203      | 94.86  |
| BRCA  | 608     | 595    | 97.86 | 586     | 96.38 | 558      | 91.78  |
| COAD  | 164     | 155    | 94.51 | 143     | 87.20 | 162      | 98.78  |
| HNSC  | 283     | 275    | 97.17 | 261     | 92.23 | 267      | 94.35  |
| KICH  | 46      | 44     | 95.65 | 44      | 95.65 | 46       | 100.00 |
| KIRC  | 303     | 302    | 99.67 | 293     | 96.70 | 301      | 99.34  |
| KIRP  | 162     | 161    | 99.38 | 158     | 97.53 | 161      | 99.38  |
| LIHC  | 212     | 192    | 90.57 | 201     | 94.81 | 186      | 87.74  |
| LUAD  | 289     | 283    | 97.92 | 282     | 97.58 | 286      | 98.96  |
| LUSC  | 277     | 272    | 98.19 | 268     | 96.75 | 275      | 99.28  |
| PRAD  | 275     | 257    | 93.45 | 260     | 94.55 | 254      | 92.36  |
| THCA  | 286     | 274    | 95.80 | 257     | 89.86 | 258      | 90.21  |
