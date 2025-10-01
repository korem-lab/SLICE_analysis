import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tcga_preds
import os

from debiasm import MultitaskDebiasMClassifier, DebiasMClassifier
from sklearn.model_selection import LeaveOneGroupOut
# from skbio.stats.composition import clr
from debiasm.torch_functions import rescale
from tcga_preds import flatten
from sklearn.metrics import roc_auc_score
from debiasm.sklearn_functions import batch_weight_feature_and_nbatchpairs_scaling
import torch

from debias_wrapper_functions import set_up_data, run_crosscenter_debiasmultitask_pairwise_eval


def make_one_analysis_run(train_nm='oncog_v1', 
                          test_nm='oncog_v1', 
                          pred_approach='CLR --> PCA --> Logreg',
                          min_n=5, 
                          min_read_counts=10
                          ):

    df_train, md_train, df_test,  md_test, all_tasks, all_centers = \
                       set_up_data(train_nm=train_nm, 
                                                    test_nm=test_nm)
    
    ## remove samples based on min read count criteria
    md_train = md_train.loc[md_train.read_count >= min_read_counts]
    md_test = md_test.loc[md_test.read_count >= min_read_counts]
    df_train=df_train.loc[md_train.index]
    df_test=df_test.loc[md_test.index]

    res_df = run_crosscenter_debiasmultitask_pairwise_eval(df_train,
                                                    md_train,
                                                    df_test, 
                                                    md_test, 
                                                    all_tasks,
                                                    all_centers,
                                                    eps_=1e-8, 
                                                    min_n=min_n
                                                    )\
                            .assign(train_ds=train_nm, 
                                    test_ds=test_nm, 
                                    datasets = '{} --> {}'.format(train_nm, 
                                                                  test_nm), 
                                    model=pred_approach, 
                                    min_n=min_n, 
                                    min_read_counts=min_read_counts
                                    )
    
    return( res_df )



def main(results_path='../results/07-debias-read-count-filtering-pairwise.csv'):

    if os.path.exists(results_path):
        raise(ValueError('Results path already exists!'))
        
    first_run=True
    for min_n in [5]:
        for min_read_counts in [10, 1000, 5000, 10000, 50000]:
            for nm in [ 'oncog_v4' ]:

                res_df = make_one_analysis_run(train_nm=nm, 
                                               test_nm=nm, 
                                               pred_approach='Multitask DEBIAS-M',
                                               min_n=min_n, 
                                               min_read_counts=min_read_counts,
                                               )

                if first_run:
                    res_df.to_csv(results_path)
                    first_run=False

                else:
                    pd.concat([pd.read_csv(results_path, index_col=0),
                               res_df]).reset_index(drop=True)\
                                             .to_csv(results_path)
    
    print('successful run completed!!')
    return(None)


if __name__=='__main__':
    main()



