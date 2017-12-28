import os
import numpy as np
import pandas as pd
import _pickle as cPickle #for pyton3.x
from sklearn.utils import shuffle
from sklearn import cross_validation
from keras.utils import to_categorical
from wx_hyperparam import WxHyperParameter
from wx_core import WxSlp, ClassifierLoocv

FEATURE_SET_DF_FILE_NAME = 'cancer12_feature_set_df.cpickle' 
TRAIN_SET_DF_FILE_NAME = 'cancer12_train_set_df.cpickle'
TCGA_ASSEM_GENE_FILE_NAME = 'GENE_LIST_TCGA_ASSEM.txt'

def StandByRow(data_frame):
    data_frame_tumor = data_frame['Tumor']
    data_frame_type = data_frame['CancerName']
    data_frame = data_frame.drop(['Tumor','CancerName'],axis=1)

    data_frame = data_frame.astype(float)
    data_frame.fillna(0, inplace=True)
    data_frame = data_frame.apply(lambda x: ((x-x.mean())/x.std()), axis=1)

    data_frame['Tumor'] = data_frame_tumor
    data_frame['CancerName'] = data_frame_type	

    return data_frame

def StandByCol(data_frame):
    data_frame_tumor = data_frame['Tumor']
    data_frame_type = data_frame['CancerName']
    data_frame = data_frame.drop(['Tumor','CancerName'],axis=1)

    data_frame = data_frame.astype(float)
    data_frame.fillna(0, inplace=True)
    data_frame = data_frame.apply(lambda x: ((x-x.mean())/x.std()), axis=0)

    data_frame['Tumor'] = data_frame_tumor
    data_frame['CancerName'] = data_frame_type	

    return data_frame

def GetValueListFromFile(file_name):
    values = []
    with open(file_name,'rt') as rFile:
        for line in rFile:
            line = line.replace('\n','')
            values.append(line)
    return values

def PreprocessTCGAassembler(data_path, feature_set_ratio):
    def walk_the_tree(dir_path):
        list_file = []
        for root, directories, files in os.walk(dir_path):
            for filename in files:
                list_file.append(filename);

        return list_file

    list_file = walk_the_tree(data_path)
    feature_df = []
    train_df = []  
    assem_gene_list = GetValueListFromFile(TCGA_ASSEM_GENE_FILE_NAME)

    for file in list_file:
        cur_type = file.split('__')[0]
        cur_df = pd.read_csv(data_path+'/'+file,sep='\t')

        print (cur_type, ' process.... ')

        #set gene symbols, erase ? symbols
        gene_id = cur_df['gene_id'].values
        genes = []
        genes_sym_only = []
        for gene in gene_id:
            symbol = gene.split('|')[0]
            genes.append(symbol)
            if symbol != '?':
                genes_sym_only.append(symbol)

        if set(assem_gene_list) == set(genes_sym_only):
            print('Genes are equal.')
        else:
            print(cur_type, ' Genes are not equal.')
            exit(1)

        cur_df['gene_id'] = genes
        cur_df = cur_df[cur_df['gene_id'] != '?']
        cur_df = cur_df.reset_index()
        cur_df = cur_df.drop('index',axis=1)
        #print(cur_df.head())

        #get tumor samples and normal sample via column names
        samples = cur_df.columns.tolist()
        tumors = []
        normals = []      
        for sample in samples:
            sample_s = sample.split('-')
            if len(sample_s) > 3:
                sample_type = sample_s[3][:2]
                if int(sample_type) < 10:#it's tumor
                    tumors.append(sample)
                else:
                    normals.append(sample)

        new_columns = genes_sym_only[:]
        new_columns.insert(0, 'Tumor')
        new_columns.insert(0, 'CancerName')        

        #get values
        df_values_tumor = []
        df_values_normal = []
        for col_name in tumors:
            cur_values = cur_df[col_name].values.tolist()
            cur_values.insert(0, True)
            cur_values.insert(0, cur_type)
            df_values_tumor.append(cur_values)
        for col_name in normals:
            cur_values = cur_df[col_name].values.tolist()
            cur_values.insert(0, False)
            cur_values.insert(0, cur_type)
            df_values_normal.append(cur_values)

        #split feature set / train set
        tumor_th = int(float(len(df_values_tumor)) * FEATURE_SET_RATIO)
        normal_th = int(float(len(df_values_normal)) * FEATURE_SET_RATIO)
        feature_df = feature_df + df_values_tumor[:tumor_th]
        feature_df = feature_df + df_values_normal[:normal_th]
        train_df = train_df + df_values_tumor[tumor_th:]
        train_df = train_df + df_values_normal[normal_th:]

    feature_df = pd.DataFrame(feature_df, columns=new_columns)
    train_df = pd.DataFrame(train_df, columns=new_columns)

    # print (feature_df.shape)
    # print (train_df.shape)

    cPickle.dump(feature_df, open(FEATURE_SET_DF_FILE_NAME,'wb'),protocol=-1)  
    cPickle.dump(train_df, open(TRAIN_SET_DF_FILE_NAME,'wb'),protocol=-1)
    print('Saving TCGA Cancer 12 type Dataframe done...')

def LoadNormFeatureSet(df, validation_ratio, RANDOM_STATE):
    df_cancer = df[df.Tumor == True]
    df_cancer = df_cancer.drop(['Tumor','CancerName'],axis=1)

    df_normal = df[df.Tumor == False]
    df_normal = df_normal.drop(['Tumor','CancerName'],axis=1)

    tumor_values = df_cancer.values.tolist()
    normal_values = df_normal.values.tolist()  

    tumor_values = shuffle(tumor_values, random_state = RANDOM_STATE)    
    normal_values = shuffle(normal_values, random_state = RANDOM_STATE)

    tumor_th = int(float(len(tumor_values)) * validation_ratio)
    normal_th = int(float(len(normal_values)) * validation_ratio)

    train_x = []
    val_x = []
    train_y = []
    val_y = []

    val_x = val_x + tumor_values[:tumor_th]
    train_x = train_x + tumor_values[tumor_th:]
    val_x = val_x + normal_values[:normal_th]
    train_x = train_x + normal_values[normal_th:]    

    for i in range(0,tumor_th):
        val_y.append(1.0)
    for i in range(tumor_th,len(tumor_values)):
        train_y.append(1.0)

    for i in range(0,normal_th):
        val_y.append(0.0)
    for i in range(normal_th,len(normal_values)):
        train_y.append(0.0)

    #shuffle
    train_x, train_y = shuffle(train_x, train_y, random_state = RANDOM_STATE)
    val_x, val_y = shuffle(val_x, val_y, random_state = RANDOM_STATE)

    #as to do 2 class softmax
    train_y = to_categorical(train_y, 2)
    val_y = to_categorical(val_y, 2)

    return np.asarray(train_x), np.asarray(train_y), np.asarray(val_x), np.asarray(val_y) 
    

def DoFeatureSelection(n_sel = 14):
    VALIDATION_RATIO = 0.2
    ITERATION = 1000
    df = cPickle.load(open(FEATURE_SET_DF_FILE_NAME,'rb'))
    df = StandByRow(df)
    print('Feature Data Frame : ',df.shape)
    gene_names = GetValueListFromFile('GENE_LIST_TCGA_ASSEM.txt')    
    feature_num = len(gene_names)
    all_weight = np.zeros(feature_num)    
    all_count = np.ones(feature_num)
    for i in range(0, ITERATION):
        train_x, train_y, val_x, val_y = LoadNormFeatureSet(df, VALIDATION_RATIO, i)
        hp = WxHyperParameter(epochs=30, learning_ratio=0.001, batch_size=32)
        sel_idx, sel_weight, val_acc = WxSlp(train_x, train_y, val_x, val_y, n_selection=min(n_sel*100, feature_num), hyper_param=hp)
        for j in range(0,min(n_sel*100, feature_num)):
            all_weight[sel_idx[j]] += sel_weight[j]
            all_count[sel_idx[j]] += 1
    all_weight = all_weight / all_count
    sort_index = np.argsort(all_weight)[::-1]    
    sel_index = sort_index[:n_sel]
    sel_weight =  all_weight[sel_index]
    gene_names = np.asarray(gene_names)
    sel_genes = gene_names[sel_index]

    return sel_index, sel_genes, sel_weight

def DoEvaluationLOOCV(sel_genes):
    RANDOM_STATE = 1
    VAL_RATIO = 0.2
    
    def DoLOOCV(all_x, all_y):
        loo = cross_validation.LeaveOneOut(len(all_x))
        acc = []
        for train_index, test_index in loo:
            train_val_x, test_x = all_x[train_index], all_x[test_index]
            train_val_y, test_y = all_y[train_index], all_y[test_index]

            train_val_x, train_val_y = shuffle(train_val_x, train_val_y, random_state=RANDOM_STATE)
            n_trn = len(train_val_x)
            n_dev = int(n_trn*VAL_RATIO)
            n_trn = n_trn - n_dev
            train_x = train_val_x[0:n_trn]
            train_y = train_val_y[0:n_trn]
            val_x = train_val_x[n_trn:]
            val_y = train_val_y[n_trn:]     

            is_correct = ClassifierLoocv(train_x, train_y, val_x, val_y, test_x, test_y)
            acc.append(is_correct)

        return acc            

    def PrintResult(cancer_type, acc):
        data = []
        for i, ct in enumerate(cancer_type):
            data.append([ct, acc[i]])

        df = pd.DataFrame(data, columns=['Type', 'Acc'])
        groupby_type = df['Acc'].groupby(df['Type'])
        print (groupby_type.describe())    
            
    type_acc = 0
    gene_names = GetValueListFromFile('GENE_LIST_TCGA_ASSEM.txt')    
    df = cPickle.load(open(TRAIN_SET_DF_FILE_NAME,'rb'))
    #df = StandByRow(df)
    sel_genes.append('Tumor')
    sel_genes.append('CancerName')
    df = df[sel_genes]

    data_label = df['Tumor'].values #tumor or normal label
    data_type = df['CancerName'].values #cancer type
    df = df.drop(['Tumor', 'CancerName'],axis=1)
    data_x = df.values #seleted features

    #let the data ordering be same
    acc_ret = DoLOOCV(data_x, data_label)
    PrintResult(data_type[:len(acc_ret)], acc_ret)

if __name__ == '__main__':    
    RAW_TCGA_DATA_FOLDER_PATH = '../TCGA_DATAS/'
    # feature set and training set be the 5:5
    FEATURE_SET_RATIO = 0.5
    # run once, to make tumor/normal feature set and training set
    if os.path.exists(FEATURE_SET_DF_FILE_NAME) == False:
        print ('DO preprocess TCGA data...')
        PreprocessTCGAassembler(RAW_TCGA_DATA_FOLDER_PATH, FEATURE_SET_RATIO)

    sel_idx, sel_genes, sel_weight = DoFeatureSelection()

    print ('\nSingle Layer WX')
    print ('selected feature index:',sel_idx)
    print ('selected feature genes:',sel_genes)
    print ('selected feature weights:',sel_weight)

    #Keras wx 14
    #sel_genes = ['EEF1A1', 'FN1', 'GAPDH', 'SFTPC', 'AHNAK', 'KLK3', 'UMOD', 'CTSB', 'COL1A1', 'GPX3', 'GNAS', 'ACTB', 'SFTPB', 'ATP1A1']

    #bioarix(naive tensorflow) wx 14
    #sel_genes = ['FN1', 'EEF1A1', 'SFTPC', 'GAPDH', 'UMOD', 'GPX3', 'FTL', 'ALB', 'P4HB', 'DCN', 'A2M', 'MGP', 'ACPP', 'CTSD']

    #Peng 14
    #sel_genes = ['KIF4A','NUSAP1','HJURP','NEK2','FANCI','DTL','UHRF1','FEN1','IQGAP3','KIF20A','TRIM59','CENPL','C16orf59','UBE2C']

    #edgeR 14    
    #sel_genes = ['LCN1','UMOD','AQP2','PATE4','SLC12A1','OTOP2','ACTN3','KRT36','ATP2A1','PRH2','AGER','PYGM','PRR4','ESRRB']

    DoEvaluationLOOCV(sel_genes)
