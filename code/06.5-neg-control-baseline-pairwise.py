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

from baseline_wrapper_functions import set_up_data, run_crosscenter_pairwise_eval


def make_one_neg_control_analysis_run(train_nm='oncog_v1', 
                                      test_nm='oncog_v1', 
                                      pred_approach='CLR --> PCA --> Logreg',
                                      min_n=5, 
                                      shuffle_type='center_and_site_shuffled_labels',
                                      filt_thresh=0.5,
                                      seed=42
                                      ):

    df_train, md_train, df_test,  md_test, all_tasks, all_centers = \
                       set_up_data(train_nm=train_nm, 
                                                    test_nm=test_nm)
    
    
    ### drop sites/center for which reshuffling in the specified group would preserve most signal 
    if shuffle_type=='center_and_site_shuffled_labels':
        sites_to_keep = md_train.groupby(['data_submitting_center_label', 
                            'tissue_source_site_label'])['disease_type']\
                            .apply(lambda x: ( x == x.mode().values[0] ).mean()<filt_thresh )


        md_train = sites_to_keep.loc[sites_to_keep].to_frame().drop('disease_type', axis=1).merge(md_train, 
                                               left_index=True, 
                                               right_on=['data_submitting_center_label', 
                                                         'tissue_source_site_label']
                                               )
        md_test = sites_to_keep.loc[sites_to_keep].to_frame().drop('disease_type', axis=1).merge(md_test, 
                                           left_index=True, 
                                           right_on=['data_submitting_center_label', 
                                                     'tissue_source_site_label']
                                           )
    elif shuffle_type=='site_shuffled_labels':
        
        sites_to_keep = md_train.groupby(['tissue_source_site_label'])['disease_type']\
                            .apply(lambda x: ( x == x.mode().values[0] ).mean()<filt_thresh )


        md_train = sites_to_keep.loc[sites_to_keep].to_frame().drop('disease_type', axis=1).merge(md_train, 
                                               left_index=True, 
                                               right_on=['tissue_source_site_label']
                                               )
        md_test = sites_to_keep.loc[sites_to_keep].to_frame().drop('disease_type', axis=1).merge(md_test, 
                                           left_index=True, 
                                           right_on=['tissue_source_site_label'] )
        
        
    elif shuffle_type=='center_shuffled_labels':
        sites_to_keep = md_train.groupby(['data_submitting_center_label'])['disease_type']\
                            .apply(lambda x: ( x == x.mode().values[0] ).mean()<filt_thresh )


        md_train = sites_to_keep.loc[sites_to_keep].to_frame().drop('disease_type', axis=1).merge(md_train, 
                                               left_index=True, 
                                               right_on=['data_submitting_center_label']
                                               )
        md_test = sites_to_keep.loc[sites_to_keep].to_frame().drop('disease_type', axis=1).merge(md_test, 
                                           left_index=True, 
                                           right_on=['data_submitting_center_label']
                                           )
        
    ## shuffle by all three possible groupings
    np.random.seed(seed)
    
    tmp = md_train.groupby(['data_submitting_center_label'] )['disease_type']\
                                                     .apply(lambda x:pd.Series( x.sample(frac=1).values, index=x.index ))
    tmp.index= tmp.index.get_level_values(-1)
    md_train['center_shuffled_labels'] = tmp
    
    
    tmp = md_train.groupby(['data_submitting_center_label',
                        'tissue_source_site_label'] )\
                                ['disease_type']\
                                    .apply(lambda x:pd.Series( x.sample(frac=1).values, 
                                                                            index=x.index ))

    tmp.index= tmp.index.get_level_values(-1)
    md_train['center_and_site_shuffled_labels'] = tmp
    
    
    
    tmp = md_train.groupby(['tissue_source_site_label'] )['disease_type']\
                                                      .apply(lambda x:pd.Series( x.sample(frac=1).values, index=x.index ))
    tmp.index= tmp.index.get_level_values(-1)
    md_train['site_shuffled_labels'] = tmp
    
    md_train['disease_type'] = md_train[shuffle_type]
    md_test = md_train.copy()

    
    res_df = run_crosscenter_pairwise_eval(df_train,
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
                                    shuffle_type=shuffle_type
                                    )
    
    return( res_df )



def main(results_path='../results/06.5-negative-control-baseline-pairwise.csv'):

    if os.path.exists(results_path):
        raise(ValueError('Results path already exists!'))
        
    first_run=True
    for min_n in [5]:
        for nm in [
                   'oncog_v4' ]:
            for shuffle_type in ['center_and_site_shuffled_labels', 
#                                  'center_shuffled_labels', 
#                                  'site_shuffled_labels', 
                                  ]:
                
                res_df = make_one_neg_control_analysis_run(train_nm=nm, 
                                                           test_nm=nm, 
                                                           pred_approach='CLR --> Logreg',
                                                           min_n=min_n, 
                                                           shuffle_type=shuffle_type
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



