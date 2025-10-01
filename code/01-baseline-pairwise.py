
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

from scipy.stats import mannwhitneyu

from sklearn.model_selection import GridSearchCV
from baseline_wrapper_functions import set_up_data, run_crosscenter_pairwise_eval, pred_func_name_map




def make_one_analysis_run(train_nm='oncog_v1', 
                          test_nm='oncog_v1', 
                          pred_approach='CLR --> PCA --> Logreg', 
                          min_n=5
                          ):

    df_train, md_train, df_test,  md_test, all_tasks, all_centers = \
                    set_up_data(train_nm=train_nm, 
                                test_nm=test_nm)
    
    res_df = run_crosscenter_pairwise_eval(df_train,
                                           md_train,
                                           df_test, 
                                           md_test, 
                                           all_tasks,
                                           all_centers,
                                           prediction_func=pred_func_name_map[pred_approach], 
                                           eps_=1e-8, 
                                           min_n=min_n)\
                    .assign(train_ds=train_nm, 
                            test_ds=test_nm, 
                            datasets = '{} --> {}'.format(train_nm, 
                                                          test_nm), 
                            model=pred_approach, 
                            min_n=min_n
                            )
    
    return( res_df )



def main(results_path='../results/01-baseline-pairwise.csv', 
         seed=42):

    if os.path.exists(results_path):
        raise(ValueError('Results path already exists!'))
    
    first_run=True
    
    for min_n in [5]: 
        np.random.seed(seed)
        for nm in [
#                    'old',
#                    'salz_24' , 
                   'oncog_v4', 
#                    'oncog_v5', 
                   'gihawi_23'
                    ]:

            res_df = make_one_analysis_run(train_nm=nm, 
                                           test_nm=nm, 
                                           pred_approach='CLR --> PCA --> Logreg', 
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
            for test_nm in [ a for a in 
                                [
#                                'salz_24' , 
#                                'oncog_v4',
                               'gihawi_23'
                                ] if a != train_nm ] :

                res_df = make_one_analysis_run(train_nm=train_nm, 
                                               test_nm=test_nm, 
                                               pred_approach='CLR --> PCA --> Logreg', 
                                               min_n=min_n
                                               )

                if first_run:
                    res_df.to_csv(results_path)
                    first_run=False

                else:
                    pd.concat([pd.read_csv(results_path, index_col=0),
                               res_df]).reset_index(drop=True)\
                                             .to_csv(results_path)   


        for train_nm in ['gihawi_23']:
            for test_nm in [ a for a in 
                                [
                               'old' , 
                               'oncog_v4'
                                ] if a != train_nm ] :

                res_df = make_one_analysis_run(train_nm=train_nm, 
                                               test_nm=test_nm, 
                                               pred_approach='CLR --> PCA --> Logreg', 
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
