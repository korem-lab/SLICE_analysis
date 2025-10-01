
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import PCA
from skbio.stats.composition import clr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from debiasm.torch_functions import rescale
from sklearn.pipeline import Pipeline

import tcga_preds

from debiasm.sklearn_functions import DebiasMClassifier, DebiasMClassifierLogAdd, \
                batch_weight_feature_and_nbatchpairs_scaling


from sklearn.model_selection import GridSearchCV

from scipy.stats import mannwhitneyu

def get_vs_random_pvalue(y_true, 
                         y_scores):
    return( 
        mannwhitneyu(y_scores[y_true==1], 
                     y_scores[y_true==0], 
                     alternative='greater'
                     ).pvalue )

def run_tuning_fit_predict(X_train, 
                           y_train,
                           groups_train,
                           X_test, 
                           md_train_1=None, 
                           train_inds=None, 
                           test_inds=None
                           ):
    
    # Define the pipeline
    pipeline = Pipeline([
        ('pca', PCA()), # PCA for dimensionality reduction
        ('log_reg', LogisticRegression(penalty='l2', 
                                       solver='newton-cg', 
                                       max_iter=100
                                      ) )  # Logistic Regression with set penalty and solver
                    ])

    # Set up the parameter grid
    param_grid = {
        'pca__n_components': [5],  # Tuning number of PCA components
        'log_reg__C':np.logspace(-4, 4, 10)  # Varying C for regularization strength
    }

    # Initialize GridSearchCV with cross-validation
    gs = GridSearchCV(estimator=pipeline, 
                       param_grid=param_grid, 
                       cv=min(5, pd.Series(y_train).value_counts().min()),
                       )
    
    gs.fit(X_train, y_train)
    if y_train.max() > 1:
        return(gs.predict_proba(X_test))
    else:
        return(gs.predict_proba(X_test)[:, 1])
    
    
pred_func_name_map = {'CLR --> PCA --> Logreg':run_tuning_fit_predict }


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

def get_site_vals(ytrain, ytest, grp_train, grp_test, site):
    train_mask = grp_train == site
    test_mask = grp_test == site
    tytr = ytrain[train_mask][ytrain[train_mask]!=-1]
    tyte = ytest[test_mask][ytest[test_mask]!=-1]
    return( pd.Series(tytr).value_counts(),
            pd.Series(tyte).value_counts() )

def mask_overlapping_site_labels_singletask(ytrain, 
                                            ytest, 
                                            grptrain, 
                                            grptest, 
                                            min_n=5
                                            ):
    
    overlapping_sites = set(grptrain[ytrain != -1]) & set(grptest[ytest != -1])

    if overlapping_sites != set():

        traincount = pd.Series( ytrain[ ytrain!=-1] ).value_counts()
        testcount = pd.Series( ytest[ ytest!=-1] ).value_counts()

        if traincount.shape[0]==2 and testcount.shape[0]==2:

            traintotals = pd.Series( ytrain[ytrain!=-1] ).value_counts()
            testtotals = pd.Series( ytest[ytest!=-1] ).value_counts()

            ## figure out if we mask the train or the test site
            ## figure out if we mask the train or the test site
            for site in pd.Series([a for a in overlapping_sites]).sort_values().values:
                        #overlapping_sites: ## since the order looping through sets is random...
                        
                site_train_counts, site_test_counts = get_site_vals(ytrain, 
                                                                    ytest, 
                                                                    grptrain,
                                                                    grptest,
                                                                    site)

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


def run_crosscenter_pairwise_eval(df_train,
                                  md_train,
                                  df_test, 
                                  md_test, 
                                  all_tasks,
                                  all_centers,
                                  prediction_func=run_tuning_fit_predict, 
                                  eps_=0,
                                  min_n=5):
    
    sort=lambda x: tuple( pd.Series(x).sort_values().values )
    all_tasks = sort([sort(l) for l in all_tasks])
    
    df_train.loc[df_train.sum(axis=1) == 0 ] += 1e-6 ## add pseudocount if needed to avoid NaN errors
    df_test.loc[df_train.sum(axis=1) == 0 ] += 1e-6
    
    all_rocs=[]
    all_eval_tasks=[]
    logo=LeaveOneGroupOut()

    all_test_centers=[]
    all_pvalues_vs_rand = []

    for iii,task in enumerate(all_tasks):
        md_train_1=md_train.loc[(md_train.disease_type.isin(task) )]
        md_test_1 =md_test.loc[md_train_1.index]
        df_train_1=df_train.loc[md_train_1.index]
        df_test_1=df_test.loc[md_train_1.index]
        

        for train_inds_, test_inds_ in logo.split(
                                    df_train_1.values, 
                                    md_train_1.disease_type.values,
                                    md_train_1.data_submitting_center_label.values
                                            ):

            test_inds = np.array( [ i for i in test_inds_ if md_test_1.disease_type.values[i] in 
                                 np.unique( md_train_1.disease_type.values[train_inds_] ) ] )

            train_inds=train_inds_.copy()


            if len( train_inds.shape )>0 and len( test_inds.shape )>0:
                if (train_inds.shape[0])>=min_n and test_inds.shape[0]>=min_n: 

                    if md_train_1.loc[md_train_1.index.values[train_inds]].disease_type.value_counts().min() >= min_n and \
                         md_train_1.loc[md_train_1.index.values[test_inds]].disease_type.value_counts().min() >= min_n:


                        tmp_df_train = rescale(df_train_1.values[train_inds]) ## add eps_ to avoid nans
                        tmp_df_test =  rescale(df_test_1.values[test_inds])

                        tmp_df_train[np.isnan(tmp_df_train)]=0
                        tmp_df_test[np.isnan(tmp_df_test)]=0

                        cols = ( df_train_1.values[train_inds] > eps_).mean(axis=0) > .01

                        X_train = clr( eps_ + tmp_df_train[:, cols] )
                        X_test  = clr( eps_ + tmp_df_test[:, cols] ) 
                        
                        y = (md_train_1.disease_type.values == task[0] ).astype(float)
                        y_train=y[train_inds]
                        y_test=y[test_inds]

                        ## remove overlap of train/test sites (or more precisely, sets them to -1)
                        y_train, y_test = mask_overlapping_site_labels_singletask(y_train, 
                                                                                  y_test, 
                                 md_train_1.tissue_source_site_label.values[train_inds],
                                 md_train_1.tissue_source_site_label.values[test_inds], 
                                                                               min_n=min_n
                                                                               )


                        X_train=X_train[y_train!=-1]
                        X_test=X_test[y_test!=-1]
                        y_train=y_train[y_train!=-1]
                        y_test=y_test[y_test!=-1]


                        ## criteria to make sure we can calculate aurocs on test set
                        if np.unique(y_test).shape[0]>1 and \
                                np.unique(y_train).shape[0]>1 and\
                                 pd.Series(y_train).value_counts().min()>=min_n and \
                                    pd.Series(y_test).value_counts().min()>=min_n:
                            
                            preds = prediction_func(X_train, 
                                                    y_train, 
                                                    md_train_1.tissue_source_site_label.values[train_inds],
                                                    X_test,
                                                    md_train, 
                                                    train_inds, 
                                                    test_inds
                                                    )

                            all_rocs.append( 
                                roc_auc_score(y_test,
                                              preds)
                                           )
                            
                            all_pvalues_vs_rand.append(
                                        get_vs_random_pvalue(y_test,  
                                                             preds) )
                                 
                            
                            all_eval_tasks.append(task)
                            all_test_centers.append(md_test_1.data_submitting_center_label[test_inds].values[0])

                            print( all_rocs )
                                

    return( pd.DataFrame({'auROC':all_rocs, 
                         'Task':[' vs '.join(a) 
                                 for a in all_eval_tasks], 
                         'Center':all_test_centers,
                          'mwu_p_vs_rand':all_pvalues_vs_rand
                         }) )