
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

def run_crosscenter_debiasmultitask_pairwise_eval(df_train,
                                                 md_train,
                                                 df_test, 
                                                 md_test, 
                                                 all_tasks,
                                                 all_centers,
                                                 eps_=0, 
                                                 min_n=5
                                                 ):

    
    
    df_train.loc[df_train.sum(axis=1)==0] += 1e-6
    df_test.loc[df_train.sum(axis=1)==0] += 1e-6
    
    all_rocs = []
    all_eval_tasks=[]
    all_test_centers = []
    all_val_run_inds = []
    all_train_sums = []
    all_test_sums = []
    avg_train_read_counts = []
    avg_test_read_counts = []
    all_test_sites = []

    np.random.seed(42)
    torch.manual_seed(42)

    md_train_1=md_train.copy()
    md_test_1 =md_test.copy().loc[md_train_1.index]
    df_train_1=df_train.copy().loc[md_train_1.index]
    df_test_1=df_test.copy().loc[md_train_1.index]
    
    mdtr = format_multitask_y(md_train, all_tasks, all_centers).loc[md_train_1.index]
    mdte = format_multitask_y(md_train, all_tasks, all_centers).loc[md_train_1.index]

    logo=LeaveOneGroupOut()
    
    for train_inds_, test_inds_ in logo.split(
                                    df_train_1.values, 
                                    md_train_1.disease_type.values,
                                    md_train_1.data_submitting_center_label.values
                                            ):

        
        ## no need to worry about any kind of train/test subsampling here, since all evaluations are
        ## only within a test-center-site
        test_inds = np.array( [ i for i in test_inds_ if md_test_1.disease_type.values[i] in 
                             np.unique( md_train_1.disease_type.values[train_inds_] ) ] )

        train_inds=train_inds_.copy()


        if len( train_inds.shape )>0 and len( test_inds.shape )>0:
            if train_inds.shape[0] >= min_n and test_inds.shape[0] >= min_n:

                tmp_df_train = rescale( df_train_1.values[train_inds])
                tmp_df_test = rescale( df_test_1.values[test_inds])
                
                tmp_df_train[np.isnan(tmp_df_train)]=eps_
                tmp_df_test[np.isnan(tmp_df_test)]=eps_

                cols = ( tmp_df_train > eps_).mean(axis=0) > .01

                X_train = tmp_df_train[:, cols]
                X_test  = tmp_df_test[:, cols] 
                
                X_train[X_train.sum(axis=1)==0]=eps_ ## theese minor pseudocounts
                X_test[X_test.sum(axis=1)==0]=eps_ ## avoid empty sample NaNs

                ## add the batch column
                X_train = np.hstack((pd.Categorical(md_train_1.data_submitting_center_label\
                                                                .values[train_inds]).codes[:, np.newaxis], 
                                     X_train))

                X_test = np.hstack(( np.array([X_train[:, 0].max()+1]*X_test.shape[0])[:, np.newaxis], 
                                 X_test))

                ## training tasks we can consider
                ccols=( pd.DataFrame(mdtr.values[train_inds]).nunique(axis=0)>2 ).values

                
                mdmc = MultitaskDebiasMClassifier(x_val = X_test, 
#                                                   batch_str = \
#                            batch_weight_feature_and_nbatchpairs_scaling(1e4, 
#                                pd.DataFrame( np.vstack([X_train, X_test]) ) ) ## default str setting
                                                  )

                mdmc.fit(X_train, mdtr.values[train_inds][:, ccols])

                ppreds = mdmc.predict_proba(X_test)

                pppreds = pd.DataFrame( np.vstack(ppreds).T,
                           columns=mdtr.columns[ccols], 
                           index=df_test_1.index.values[test_inds]
                           )

                test_set=mdtr.values[test_inds][:, ccols]

                for j in range(test_set.shape[1]):
                    ttt_inds=test_set[:, j]!=-1
                    pqwe=pd.Series( test_set[:, j][ttt_inds] ).value_counts()


                    if pqwe.min()>=min_n \
                        and ( pqwe.shape[0]==2\
                            and pd.Series( mdtr.values[:, ccols][:, j][train_inds] )\
                                .value_counts().min()>=min_n):

                        for test_site in np.unique(md_test.tissue_source_site_label.values[test_inds]):
                            test_site_mask=md_test.tissue_source_site_label.values[test_inds]==test_site

                            ttt_inds_tmps = ttt_inds * test_site_mask
                            pqwe_tmp =pd.Series(  test_set[:, j][ttt_inds_tmps] ).value_counts()

                            if pqwe_tmp.min()>=min_n \
                                and ( pqwe_tmp.shape[0]==2):



                                all_rocs.append( roc_auc_score(test_set[:, j][ttt_inds_tmps],
                                                                pppreds.values[:, j][ttt_inds_tmps] 
                                                               ) )

                                all_test_sites.append(test_site)

                                all_test_centers.append(md_test_1.data_submitting_center_label.values\
                                                                [test_inds][0])

                                all_eval_tasks.append(mdte.columns[ccols][j])

                                all_train_sums.append(' - '.join([str(a) for a in 
                                                    pd.Series( mdtr.values[:, ccols][:, j][train_inds] )\
                                                            .value_counts().values] ))
                                all_test_sums.append(' - '.join([str(a) for a in pqwe_tmp.values]) )


                                avg_train_read_counts.append( 
                                   ' - '.join( [str( md_train_1.read_count.values[train_inds]\
                                                           [ mdtr.values[:, ccols][:, j][train_inds] == labval \
                                                               ].mean() )
                                                for labval in [0,1]
                                               ] ) )

                                avg_test_read_counts.append(
                                    ' - '.join( [str( md_test_1.read_count.values[test_inds]\
                                                         [ mdtr.values[:, ccols][:, j][test_inds] == labval \
                                                               ].mean() )
                                                for labval in [0,1]
                                               ] ) )

                print(all_rocs)
                    

    return(pd.DataFrame({'auROC':all_rocs, 
                           'Task':[ a for a in all_eval_tasks], 
                           'Center':all_test_centers, 
                           'train_sums':all_train_sums, 
                           'test_sums':all_test_sums,
                           'train_read_counts':avg_train_read_counts,
                           'test_read_counts':avg_test_read_counts, 
                           'test_site':all_test_sites
                           })

          )



def make_one_analysis_run(train_nm='oncog_v1', 
                          test_nm='oncog_v1', 
                          pred_approach='CLR --> PCA --> Logreg',
                          min_n=5
                          ):

    df_train, md_train, df_test,  md_test, all_tasks, all_centers = \
                       set_up_data(train_nm=train_nm, 
                                                    test_nm=test_nm)



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
                                    min_n=min_n
                                    )
    
    return( res_df )






def main(results_path='../results/05-debias-pairwise-eval-within-site.csv'):

    if os.path.exists(results_path):
        raise(ValueError('Results path already exists!'))
        
    first_run=True
    for min_n in [5]:
        for nm in [
#                    'old',
#                    'salz_24' , 
                   'oncog_v4', 
#                    'oncog_v5', 
#                    'gihawi_23',
         ]:

            res_df = make_one_analysis_run(train_nm=nm, 
                                           test_nm=nm, 
                                           pred_approach='Multitask DEBIAS-M',
                                           min_n=min_n
                                           )

            if first_run:
                res_df.to_csv(results_path)
                first_run=False

            else:
                pd.concat([pd.read_csv(results_path, index_col=0),
                           res_df]).reset_index(drop=True)\
                                         .to_csv(results_path)   

#         for train_nm in ['old', 'oncog_v4']:
#             for test_nm in [
#                             'salz_24' , 
#                             'oncog_v4', 
#                             'gihawi_23'
#                             ]:
#                 if train_nm != test_nm:

#                     res_df = make_one_analysis_run(train_nm=train_nm, 
#                                                    test_nm=test_nm, 
#                                                    pred_approach='Multitask DEBIAS-M',
#                                                    min_n=min_n
#                                                    )

#                     if first_run:
#                         res_df.to_csv(results_path)
#                         first_run=False

#                     else:
#                         pd.concat([pd.read_csv(results_path, index_col=0),
#                                    res_df]).reset_index(drop=True)\
#                                                  .to_csv(results_path)   



    print('successful run completed!!')
    return(None)


if __name__=='__main__':
    main()