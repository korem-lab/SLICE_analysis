import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tcga_preds
import os


from debiasm import MultitaskDebiasMClassifier
from sklearn.model_selection import LeaveOneGroupOut
# from skbio.stats.composition import clr
from debiasm.torch_functions import rescale
from tcga_preds import flatten
from sklearn.metrics import roc_auc_score
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

    ## align indices across the different tables
    qq=pd.Series( list(df_train.index) + list(df_test.index) ).value_counts()

    qq=pd.Series( list(df_train.index) + list(df_test.index) +\
                       list(md_test.index)  ).value_counts()
    inds=qq.loc[qq>2]
    df_train, md_train, df_test, md_test = \
            [a.loc[inds.index] for a in 
                  [df_train, md_train, df_test, md_test]  ]
    
    all_tasks, all_centers = tcga_preds.prep_pairwise_tasks(md_train)
    
    if train_nm != test_nm:
        ## some names are g__..., others are just genus
        
        ## align the two tables' feature spaces
        df_train.columns = \
                df_train.columns.str.split('g__').str[-1]
        
        df_test.columns = \
                df_test.columns.str.split('g__').str[-1]


        for a in df_test.columns:
            if a not in df_train.columns:
                df_train[a]=0 ### initialize empty columns to prepare the aligning of features
        
        ## align train features to test set
        df_train=df_train[df_test.columns]
        
    
    return(df_train, 
           md_train, 
           df_test, 
           md_test, 
           all_tasks, 
           all_centers
           )



def format_multitask_y(md, all_tasks):
    vvs = []
    for task in all_tasks:
        vvs.append( ( md['disease_type'] == task[1] )*2 + \
                        ( md['disease_type'] == task[0] )*1 - 1 )
        
        
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
            
            ## if we only have one type in the task/center, we can just mask
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

def run_crosscenter_debiasmultitask_onevall_eval(df_train,
                                                 md_train,
                                                 df_test, 
                                                 md_test, 
                                                 all_tasks,
                                                 eps_=0, 
                                                 min_n=5,
                                                 ):
    
    sort=lambda x: tuple( pd.Series(x).sort_values().values )
    all_tasks = sort([sort(l) for l in all_tasks])[::-1]
    
    ## add pseudocount if needed to avoid NaN errors
    df_train.loc[df_train.sum(axis=1)==0] += 1e-6
    df_test.loc[df_train.sum(axis=1) ==0] += 1e-6
    
    all_rocs = []
    all_eval_tasks=[]
    all_test_centers = []
    all_pvalues_vs_rand = []
    all_inference_dfs = []

    np.random.seed(42)
    torch.manual_seed(42)

    md_train_1=md_train.copy()
    md_test_1 =md_test.copy().loc[md_train_1.index]
    df_train_1=df_train.copy().loc[md_train_1.index]
    df_test_1=df_test.copy().loc[md_train_1.index]
    
    mdtr_ = format_multitask_y(md_train, all_tasks).loc[md_train_1.index]
    colvals=mdtr_.columns
    mdtr_=mdtr_.values
    mdte_ = format_multitask_y(md_train, all_tasks).loc[md_train_1.index].values


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

        mdtr[train_inds], mdtr[test_inds] = mask_overlapping_site_labels(
                                             mdtr[train_inds].copy(),
                                             mdtr[test_inds].copy(),
                                             md_train.tissue_source_site_label.values[train_inds].copy(),
                                             md_train.tissue_source_site_label.values[test_inds].copy(), 
                                             min_n=min_n
                                             )

        mdte=mdtr

        if len( train_inds.shape )>0:
            if train_inds.shape[0] >= min_n and test_inds.shape[0] >= min_n:
                
                tmp_df_train = rescale(eps_+df_train_1.values[train_inds])
                tmp_df_test = rescale(eps_+df_test_1.values[test_inds])

                cols = ( tmp_df_train > eps_).mean(axis=0) > .01
                
#                 print(df_train_1.columns.values[cols])
                
                X_train = tmp_df_train[:, cols]
                X_test  = tmp_df_test[:, cols] 


                ## add the batch column
                X_train = np.hstack((pd.Categorical(md_train_1.data_submitting_center_label\
                                                                .values[train_inds]).codes[:, np.newaxis], 
                                     X_train))

                X_test = np.hstack(( np.array([X_train[:, 0].max()+1]*X_test.shape[0])[:, np.newaxis], 
                                 X_test))

                ## > 2 since there will be some `-1` unknowns
                ccols=( pd.DataFrame(mdtr[train_inds]).nunique(axis=0)>2 ).values 
                mdmc = MultitaskDebiasMClassifier(x_val = X_test, 
#                                                    batch_str = \
#                            batch_weight_feature_and_nbatchpairs_scaling(1e4, ## default str setting
#                                                     pd.DataFrame( np.vstack([X_train, X_test])
#                                                                 ))
                                                  )
                mdmc.fit(X_train, mdtr[train_inds][:, ccols])
                ppreds = mdmc.predict_proba(X_test)

                pppreds = pd.DataFrame( np.vstack(ppreds).T,
                           columns=colvals[ccols], 
                           index=df_test_1.index.values[test_inds]
                           )
                
#                 print(ccols)

                formatted_preds = \
                        pd.DataFrame({
                            tt: np.vstack( [ [ 1-pppreds.values[:, i], 
                                            pppreds.values[:, i] ][np.where( np.array(a)==tt )[0][0]] 
                                          for i,a in enumerate( pppreds.columns.values ) 
                                        if tt in a] ).mean(axis=0) 
                                     for tt in np.unique( flatten( pppreds.columns.values ) ) 
                                    }, 
                                    index=pppreds.index.values
                                    )
                
#                 for j in range(P):
                for i,a in enumerate( colvals[ccols] ):
                    summary_dict = \
                           {'coefs': np.array([-1, 1]) @ 
                                 [p for p in mdmc.model.linear_weights[i].parameters()][0].detach().numpy(), 
                                 'feature_name':df_train_1.columns.values[cols],
                                  'test_center':md_test_1.data_submitting_center_label.values\
                                           [test_inds][0],
                                 'eval_task':'-'.join(a), 
                                 }
                    
                    coef_summary_df=pd.DataFrame(summary_dict)
                    all_inference_dfs.append(coef_summary_df)
                      
                

                formatted_preds = pd.DataFrame(
                                               rescale(formatted_preds.values),
                                               columns=formatted_preds.columns,
                                               index=formatted_preds.index
                                               )

                qq=pd.get_dummies( md_test_1.disease_type[test_inds] )
                test_formatted = qq[[q for q in formatted_preds.columns if q in qq.columns]]
                formatted_preds=formatted_preds[test_formatted.columns]

                ## Mask the corresponding test samples
                ll = test_formatted.values.astype(int)
                ll[ (mdtr[test_inds]==-1).mean(axis=1)==1 ] = -1
                test_formatted = pd.DataFrame(ll, 
                                              index=test_formatted.index, 
                                              columns=test_formatted.columns
                                              )
                
                
                
                

                for j in range(formatted_preds.shape[1]):
                    if (test_formatted.iloc[:, j][ test_formatted.iloc[:, j]!=-1 ]).sum()>=min_n:

                            all_rocs.append( roc_auc_score(
                                        test_formatted.iloc[:, j].values[ test_formatted.iloc[:, j].values!=-1 ], 
                                       formatted_preds.iloc[:, j].values[ test_formatted.iloc[:, j].values!=-1 ]
                                                           ) )
                            
                            all_pvalues_vs_rand.append(
                                        get_vs_random_pvalue(
                            test_formatted.iloc[:, j].values[ test_formatted.iloc[:, j].values!=-1 ], 
                            formatted_preds.iloc[:, j].values[ test_formatted.iloc[:, j].values!=-1 ] ) 
                                                    )
                            
                            all_test_centers.append(md_test_1.data_submitting_center_label.values\
                                                            [test_inds][0])
                            all_eval_tasks.append(formatted_preds.columns[j])
                            
                            
#                             summary_dict = \
#                                         {'coefs': np.array([-1, 1]) @ 
#                                               [p for p in mdmc.model.linear_weights[1].parameters()][j].detach().numpy(), 
#                                               'feature_name':df_train_1.columns.values[cols],
#                                                'test_center':md_test_1.data_submitting_center_label.values\
#                                                         [test_inds][0],
#                                               'eval_task':colvals[ccols][j], 
#                                               }
                            
#                             for a in summary_dict:
#                                 print(a)
#                                 print(np.array(summary_dict[a]).shape)
                            
                            
#                             coef_summary_df = \
#                                 pd.DataFrame({'coefs': np.array([-1, 1]) @ 
#                                               [p for p in mdmc.model.linear_weights[1].parameters()][j].detach().numpy(), 
#                                               'feature_name':df_train_1.columns.values[cols],
#                                                'test_center':md_test_1.data_submitting_center_label.values\
#                                                         [test_inds][0],
#                                               'eval_task':colvals[ccols][j], 
#                                               })
                        
#                             all_inference_dfs.append(coef_summary_df)
                            

                print(all_rocs)

                
#     pd.concat(all_inference_dfs).to_csv('../results/COEF_ONE_V_ALL_DEBIAS_INFERENCE.csv')
#     pd.concat(all_inference_dfs).to_csv('../results/COEF_ONE_V_ALL_DEBIAS_zebra.csv')
    
    return(pd.DataFrame({'auROC':all_rocs, 
                           'Task':[ a for a in all_eval_tasks], 
                           'Center':all_test_centers,
                           'mwu_p_vs_rand':all_pvalues_vs_rand
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



    res_df = run_crosscenter_debiasmultitask_onevall_eval(df_train,
                                                    md_train,
                                                    df_test, 
                                                    md_test, 
                                                    all_tasks,
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



def main(results_path='../results/04-debias-onevall.csv'):

    if os.path.exists(results_path):
        raise(ValueError('Results path already exists!'))
        
    seed=42
    
    first_run=True
    
    for min_n in [5]:
        for nm in [
                   'old',
                   'salz_24' , 
                   'oncog_v4', 
                   'oncog_v5', 
        ]:
            
            np.random.seed(seed)
            torch.manual_seed(seed)

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

        for train_nm in ['old', 'oncog_v4']:
            for test_nm in [ 'salz_24' , 
                             'oncog_v4', 
                           ]:
                
                if train_nm != test_nm:
                    res_df = make_one_analysis_run(train_nm=train_nm, 
                                                   test_nm=test_nm, 
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
                        
        for test_nm in [
                         'old', 
                         'oncog_v4' 
                           ]:
            for train_nm in [a for a in  [
                                          'salz_24' , 
                                          'oncog_v4', 
                                            ]
                                         ]:
                if train_nm != test_nm:
                    res_df = make_one_analysis_run(train_nm=train_nm, 
                                                   test_nm=test_nm, 
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


    print('successful run completed!!')
    return(None)


if __name__=='__main__':
    main()


