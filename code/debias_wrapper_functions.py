
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tcga_preds
import os

from debiasm import MultitaskDebiasMClassifier, DebiasMClassifier
from sklearn.model_selection import LeaveOneGroupOut
from debiasm.torch_functions import rescale
from tcga_preds import flatten
from sklearn.metrics import roc_auc_score, average_precision_score
from debiasm.sklearn_functions import batch_weight_feature_and_nbatchpairs_scaling
import torch

from scipy.stats import mannwhitneyu

def get_vs_random_pvalue(y_true, 
                         y_scores):
    return( 
        mannwhitneyu(y_scores[y_true==1], 
                     y_scores[y_true==0], 
                     alternative='greater'
                     ).pvalue )

def set_up_data(train_nm='oncog_v1', 
                test_nm='oncog_v1'):
    
    df_train, md_train = tcga_preds.load_data(train_nm,
                                              use_WGS=True, 
                                              )
    df_test, md_test = tcga_preds.load_data(test_nm,
                                            use_WGS=True, 
                                            )

    qq=pd.Series( list(df_train.index) + list(df_test.index) ).value_counts()

    qq=pd.Series( list(df_train.index) + list(df_test.index) +\
                       list(md_test.index)  ).value_counts()
    inds=qq.loc[qq>2]

    df_train, md_train, df_test, md_test = \
            [a.loc[inds.index] for a in 
                  [df_train, md_train, df_test, md_test]  ]
    
    all_tasks, all_centers = tcga_preds.prep_pairwise_tasks(md_train)
    
    if train_nm != test_nm:
        ## align the two tables' feature spaces
        df_train.columns = \
                df_train.columns.str.split('g__').str[-1]
        
        df_test.columns = \
                df_test.columns.str.split('g__').str[-1]


        for a in df_test.columns:
            if a not in df_train.columns:
                df_train[a]=0

        df_train=df_train[df_test.columns]
    
    return(df_train, 
           md_train, 
           df_test, 
           md_test, 
           all_tasks, 
           all_centers
           )


def format_multitask_y(md, all_tasks, all_centers):
    vvs = []
    for i,task in enumerate(all_tasks):
        vvs.append( ( md['disease_type'] == task[1] )*2 + \
                        ( md['disease_type'] == task[0] )*1 - 1 )
        vvs[-1][md['data_submitting_center_label'].isin(all_centers[i]==False)]=-1
        
        
    return( pd.DataFrame(np.vstack( vvs ).T, 
                         index=md.index, 
                         columns=all_tasks
                         ) )


def get_site_vals(ytrain, ytest, grp_train, grp_test, site, task):
    train_mask = grp_train == site
    test_mask = grp_test == site
    tytr = ytrain[train_mask, task][ytrain[train_mask, task]!=-1]
    tyte = ytest[test_mask, task][ytest[test_mask, task]!=-1]
    return( pd.Series(tytr).value_counts(),
            pd.Series(tyte).value_counts())

def mask_overlapping_site_labels(ytrain, 
                                 ytest, 
                                 grptrain, 
                                 grptest, 
                                 min_n=5
                                 ):
    
    for task in range(ytrain.shape[1]):

        overlapping_sites = set(grptrain[ytrain[:, task] != -1]) & set(grptest[ytest[:, task] != -1])
        
        if overlapping_sites != set():

            traincount = pd.Series( ytrain[:, task][ ytrain[:, task]!=-1] ).value_counts()
            testcount = pd.Series( ytest[:, task][ ytest[:, task]!=-1] ).value_counts()

            ## if we only have one type in the task/center, we can just mask the whole task
            if testcount.shape[0] == 1:
                ytest[:, task]=-1

            if traincount.shape[0]==1:
                ytrain[:, task]=-1

            if traincount.shape[0]==2 and testcount.shape[0]==2:


                traintotals = pd.Series( ytrain[:, task][ytrain[:, task]!=-1] ).value_counts()
                testtotals = pd.Series( ytest[:, task][ytest[:, task]!=-1] ).value_counts()

                ## figure out if we mask the train or the test site
                for site in pd.Series([a for a in overlapping_sites]).sort_values().values:
                                #overlapping_sites: ## since the order looping through sets is random...
                    site_train_counts, site_test_counts = get_site_vals(ytrain, 
                                                                        ytest, 
                                                                        grptrain,
                                                                        grptest,
                                                                        site, 
                                                                        task)

                    train_mask = grptrain == site
                    test_mask = grptest == site

                    possible_train_counts = traintotals.subtract(site_train_counts, fill_value=0)
                    possible_test_counts = testtotals.subtract(site_test_counts, fill_value=0)
                            
                     ## check how many datapoints we lose, mask labels as needed based on that
                    if site_test_counts.sum()>site_train_counts.sum():
                        if possible_train_counts.min() >= min_n:
                            ytrain[train_mask] = -1
                            traintotals=possible_train_counts

                        else:
                            ytest[test_mask] = -1
                            testtotals=possible_test_counts
                    else:
                        if possible_test_counts.min() >= min_n:
                            ytest[test_mask] = -1
                            testtotals=possible_test_counts
                        else:
                            ytrain[train_mask] = -1
                            traintotals=possible_train_counts
                        
    return(ytrain, ytest)


def run_crosscenter_debiasmultitask_pairwise_eval(df_train,
                                                  md_train,
                                                  df_test, 
                                                  md_test, 
                                                  all_tasks,
                                                  all_centers,
                                                  eps_=0, 
                                                  min_n=5, 
                                                  seed=42
                                                  ):
    
    sort=lambda x: tuple( pd.Series(x).sort_values().values )
    all_tasks = sort([sort(l) for l in all_tasks])
    
    df_train.loc[df_train.sum(axis=1)== 0] += 1e-8 ## add pseudocount if needed to avoid NaN errors
    df_test.loc[df_test.sum(axis=1) == 0] += 1e-8 ## < 10
        
    all_rocs = []
    all_eval_tasks=[]
    all_test_centers = []
    all_val_run_inds = []
    all_train_sums = []
    all_test_sums = []
    avg_train_read_counts = []
    avg_test_read_counts = []
    all_pvalues_vs_rand = []
    all_inference_dfs = []
    all_auprs = []
    class_balances=[]

    np.random.seed(seed)
    torch.manual_seed(seed)

    ## use traing set metadata as index reference for all dataframes
    md_train_1=md_train.copy()
    md_test_1 =md_test.copy().loc[md_train_1.index]
    df_train_1=df_train.copy().loc[md_train_1.index]
    df_test_1=df_test.copy().loc[md_train_1.index]
    
    mdtr_ = format_multitask_y(md_train, all_tasks, all_centers).loc[md_train_1.index]
    colvals=mdtr_.columns
    mdtr_=mdtr_.values
    mdte_ = format_multitask_y(md_train, all_tasks, all_centers).loc[md_train_1.index].values

    logo=LeaveOneGroupOut()
    
    for train_inds_, test_inds_ in logo.split(
                                    df_train_1.values, 
                                    md_train_1.disease_type.values,
                                    md_train_1.data_submitting_center_label.values
                                            ):


        test_inds = np.array( [ i for i in test_inds_ if md_test_1.disease_type.values[i] in 
                             np.unique( md_train_1.disease_type.values[train_inds_] ) ] )

        train_inds=train_inds_.copy()


        mdtr = mdtr_.copy()

        if len( train_inds.shape )>0 and len( test_inds.shape )>0:
            if train_inds.shape[0] >= min_n and test_inds.shape[0] >= min_n:
                
                mdtr[train_inds], mdtr[test_inds] = mask_overlapping_site_labels(
                                             mdtr[train_inds].copy(),
                                             mdtr[test_inds].copy(),
                                             md_train.tissue_source_site_label.values[train_inds].copy(),
                                             md_train.tissue_source_site_label.values[test_inds].copy(), 
                                             min_n=min_n
                                             )

                mdte=mdtr


                tmp_df_train = rescale(df_train_1.values[train_inds])
                tmp_df_test =  rescale( df_test_1.values[test_inds]) 

                cols = ( tmp_df_train > eps_).mean(axis=0) > .01
                
                tmp_df_train[np.isnan(tmp_df_train)]=eps_
                tmp_df_test[np.isnan(tmp_df_test)]=eps_
                
                X_train = tmp_df_train[:, cols]
                X_test  = tmp_df_test[:, cols]
                
                X_train[X_train.sum(axis=1)==0]=eps_ ## theese minor pseudocounts
                X_test[X_test.sum(axis=1)==0]=eps_ ##     avoid empty sample NaNs

                ## add the batch column
                X_train = np.hstack((pd.Categorical(md_train_1.data_submitting_center_label\
                                                                .values[train_inds]).codes[:, np.newaxis], 
                                     X_train))

                X_test = np.hstack(( np.array([X_train[:, 0].max()+1]*X_test.shape[0])[:, np.newaxis], 
                                 X_test))


                ## training tasks we can consider
                ccols=( pd.DataFrame(mdtr[train_inds]).nunique(axis=0)>2 ).values


                mdmc = MultitaskDebiasMClassifier(x_val = X_test)

                mdmc.fit(X_train, mdtr[train_inds][:, ccols])

                ppreds = mdmc.predict_proba(X_test)

                pppreds = pd.DataFrame( np.vstack(ppreds).T,
                           columns=colvals[ccols], 
                           index=df_test_1.index.values[test_inds]
                           )

                test_set=mdtr[test_inds][:, ccols]

                for j in range(test_set.shape[1]):
                    ttt_inds=test_set[:, j]!=-1
                    pqwe=pd.Series( test_set[:, j][ttt_inds] ).value_counts()


                    if pqwe.min()>=min_n \
                        and ( pqwe.shape[0]==2\
                            and pd.Series( mdtr[:, ccols][:, j][train_inds] )\
                                .value_counts().min()>=min_n):

                        all_rocs.append( roc_auc_score(test_set[:, j][ttt_inds],
                                                       pppreds.values[:, j][ttt_inds] 
                                                       ) )
                        
                        class_balances.append(test_set[:, j][ttt_inds].mean())
                        
                        all_auprs.append( average_precision_score(test_set[:, j][ttt_inds],
                                                                   pppreds.values[:, j][ttt_inds] 
                                                                   ) )
                        
                        all_pvalues_vs_rand.append(
                                        get_vs_random_pvalue( test_set[:, j][ttt_inds],
                                                                pppreds.values[:, j][ttt_inds] ) 
                                                    ) 
                        
                        all_test_centers.append(md_test_1.data_submitting_center_label.values\
                                                        [test_inds][0])
                        all_eval_tasks.append(colvals[ccols][j])
                        all_val_run_inds.append(True)

                        all_train_sums.append(' - '.join([str(a) for a in 
                                            pd.Series( mdtr[:, ccols][:, j][train_inds] )\
                                                    .value_counts().values] ))
                        all_test_sums.append(' - '.join([str(a) for a in pqwe.values]) )


                        avg_train_read_counts.append( 
                           ' - '.join( [str( md_train_1.read_count.values[train_inds]\
                                                   [ mdtr[:, ccols][:, j][train_inds] == labval \
                                                      ].sum() )
                                        for labval in [0,1]
                                       ] ) )

                        avg_test_read_counts.append(
                            ' - '.join( [str( md_test_1.read_count.values[test_inds]\
                                                 [ mdtr[:, ccols][:, j][test_inds] == labval \
                                                       ].sum() ) #].mean() )
                                        for labval in [0,1]
                                       ] ) )
                        
                        

                print(all_rocs)
                
                    
    return(pd.DataFrame({'auROC':all_rocs, 
                         'Task':[ a for a in all_eval_tasks], 
                         'Center':all_test_centers, 
                         'filtered_train_or_test':all_val_run_inds, 
                         'train_sums':all_train_sums, 
                         'test_sums':all_test_sums,
                         'train_read_counts':avg_train_read_counts,
                         'test_read_counts':avg_test_read_counts, 
                         'mwu_p_vs_rand':all_pvalues_vs_rand, 
                         'auPR':all_auprs, 
                         'Class balance':class_balances
                         })

          )