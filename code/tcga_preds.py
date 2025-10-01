import numpy as np
import pandas as pd

def flatten(xss):
    return [x for xs in xss for x in xs]


## dictionary outlining different tables
path_dict = {'old':   {'data':'../data/original-2020/Kraken-TCGA-Raw-Data-17625-Samples.csv', 
                       'metadata':'../data/original-2020/Metadata-TCGA-Kraken-17625-Samples.csv'},
             'oncog_v1':{'data':'../data/oncogene-2024/TableS8_T2T_KrakenUniq_BIO_Fullset.csv', 
                       'metadata':'../data/oncogene-2024/TableS9_metadata_KrakenUniq_BIO_Fullset.csv'},
             'oncog_v2':{'data':'../data/oncogene-2024/TableS10_KrakenUniq_BIO_HG38_intersected.csv', 
                       'metadata':'../data/oncogene-2024/TableS9_metadata_KrakenUniq_BIO_Fullset.csv'},
             'oncog_v3':{'data':'../data/oncogene-2024/TableS11_KrakenUniq_BIO_T2T_intersected.csv', 
                       'metadata':'../data/oncogene-2024/TableS9_metadata_KrakenUniq_BIO_Fullset.csv'},
             'oncog_v4':{'data':'../data/oncogene-2024/TableS12_KrakenUniq_BIO_Pan_intersected.csv', 
                       'metadata':'../data/oncogene-2024/TableS9_metadata_KrakenUniq_BIO_Fullset.csv'},
             'oncog_v5':{'data':'../data/oncogene-2024/TableS13_RS210clean_filtered_abundances.csv', 
                       'metadata':
                           '../data/oncogene-2024/TableS14_metadata_RS210clean_filtered_abundances.csv'},
             'salz_24':{'data':'../data/salzberg-2024/salz-processing.csv',
                        'metadata':'../data/oncogene-2024/TableS9_metadata_KrakenUniq_BIO_Fullset.csv'
                       },
             'zebra_0':{'data':'../data/greg-2025/counts_wgs.csv', 
                       'metadata':
                           '../data/greg-2025/metadata_for_counts_full_merged_table_FOR_TAL_7Feb25.csv'},
             'zebra_25':{'data':'../data/greg-2025/counts_wgs.csv', 
                       'metadata':
                           '../data/greg-2025/metadata_for_counts_full_merged_table_FOR_TAL_7Feb25.csv'},
              'zebra_50':{'data':'../data/greg-2025/counts_wgs.csv', 
                       'metadata':
                           '../data/greg-2025/metadata_for_counts_full_merged_table_FOR_TAL_7Feb25.csv'},
              'zebra_75':{'data':'../data/greg-2025/counts_wgs.csv', 
                       'metadata':
                           '../data/greg-2025/metadata_for_counts_full_merged_table_FOR_TAL_7Feb25.csv'},
               'zebra_90':{'data':'../data/greg-2025/counts_wgs.csv', 
                       'metadata':
                           '../data/greg-2025/metadata_for_counts_full_merged_table_FOR_TAL_7Feb25.csv'},
              'gihawi_23':{'data':'../data/gihawi-2023/concatenated-gihawi-data.csv', 
                       'metadata':
                           '../data/greg-2025/metadata_for_counts_full_merged_table_FOR_TAL_7Feb25.csv'},
            }





def load_data(dataset_name, 
              use_WGS=True, 
              path_dict=path_dict
              ):
    
    df=pd.read_csv(path_dict[dataset_name]['data'], index_col=0)
    md = pd.read_csv(path_dict[dataset_name]['metadata'],index_col=0)#.loc[df.index]
    md = md.loc[ md.sample_type == 'Primary Tumor' ] ## sticking to primary
    
     
    if 'zebra' in dataset_name: ## 
        zebra = pd.read_csv('../data/greg-2025/zebra_coverages_RS210_FOR_TAL_7Feb25.csv', index_col=0)
        coverage_frac = float( dataset_name.split('_')[-1] )/100. ## fraction threshold based on input name
        df=df.loc[:, zebra.loc[df.columns].coverage_ratio > coverage_frac ]
        lineages = pd.read_csv('../data/greg-2025/RS210_lineages_formatted.csv', index_col=0)\
                                    .loc[df.columns]
        df = df.loc[df.sum(axis=1)>10]
        df.columns=lineages.species.values
        md=md.loc[md.data_submitting_center_label.isna()==False]
    
    if dataset_name == 'gihawi_23':
        md=md.set_index('knightlabID')
        
    else:    
        if dataset_name=='old':
            ### remove 'putative contaminant' columns
            cols = pd.read_csv('../data/original-2020/Kraken-TCGA-Voom-SNM-All-Putative-Contaminants-Removed-Data.csv', 
                        index_col=0, nrows=1).columns
            df = df.loc[:, cols[cols.str.contains('contaminant')==False]]

            md=md.loc[md.platform=='Illumina HiSeq']
        else:
            md=md.loc[md.cgc_platform=='Illumina HiSeq']

            if dataset_name != 'salz_24': ## use `knightlabID` as index to allow for aligning across datasets
                df = df.loc[df.index.isin(md.index)]
                md = md.loc[md.index.isin(df.index)].loc[df.index]
                df=df.loc[md.index]
                md=md.set_index('knightlabID')

            else:

                md=md.loc[md.knightlabID.isna()==False]
                md=md.set_index('knightlabID')

                df = df.loc[df.index.isin(md.index)]
                md = md.loc[md.index.isin(df.index)].loc[df.index]
                df=df.loc[md.index]

            df.index=md.index
            md=md.loc[md.index.isna()==False]
            df=df.loc[md.index]

        if use_WGS:
            md=md.loc[md.experimental_strategy=='WGS'] ## in general we're sticking with WGS

    ## align index of data and metadata
    df = df.loc[df.index.isin(md.index)]
    md = md.loc[md.index.isin(df.index)].sort_values('disease_type')
             ## this sorting just makes the order of tasks in result tables consistent across datasets
    df = df.loc[md.index]
    md['read_count'] = df.sum(axis=1)
    
    print(df.shape)
    print(md.shape)
    return(df, md)


def prep_pairwise_tasks(md):
    ## look for disease pairs across centers that we can both model/test
    unique_pairs = flatten( [[ a for a in zip( md.disease_type.unique(), 
                     md.disease_type.unique()[i:] )]
     for i in range(1, md.disease_type.unique().shape[0]) ] )


    all_tasks=[]
    task_centers = []
    for task in unique_pairs:
        md_tmp = md.loc[md.disease_type.isin(task)] ## find centers that have samples from both tasks
        dd=md_tmp.groupby('data_submitting_center_label')['disease_type'].nunique()
        dd=dd.loc[dd>1] ## no use in centers w/ one disease type
    
        if (dd.shape[0]>1):
            all_tasks.append(task)
            task_centers.append(dd.index)
            
    return(all_tasks, task_centers) ## list of pairwise tasks, and relevant centers

