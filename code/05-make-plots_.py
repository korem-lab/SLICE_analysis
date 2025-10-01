import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

sns.set(rc={'axes.facecolor':'white', 
            'figure.facecolor':'white', 
            'axes.edgecolor':'black', 
            'grid.color': 'black', 
            'axes.grid': False
            }, 
        
        style='ticks',
        font_scale=2
        )


def make_boxplot(df,
                 dataset = 'oncog_v4 --> oncog_v4',
                 min_n=5,
                 y_val='auROC',
                 x_val='Task',
                 col='lightblue'
                 ):
    
    if x_val!='datasets':
        plot_df = df.loc[(df.datasets==dataset )&\
                         (df.min_n==min_n)].sort_values('auROC', 
                                                        ascending=False)
    else:
        plot_df = df.loc[(df.min_n==min_n)]

    
    ax=sns.boxplot(x=x_val, 
                   y=y_val, 
                   data=plot_df, 
                   color=col,#'lightblue', 
                   fliersize=0
                )
    sns.swarmplot(x=x_val, 
                  y=y_val, 
                  data=plot_df, 
                  color='black', 
                  s=5,
                  ax=ax
                  )
    plt.ylim(0,1)
    plt.xticks(rotation=90)
    return(ax)



def main(results_path='../results/', 
         hide_axes=False):
    
    ### Fig 1a
    df=pd.read_csv('../results/debias-pairwise.csv', index_col=0)\
                        .groupby(['Task','Center', 'datasets']).head(1)
    
    plt.figure(figsize=(12,8))
    ax=make_boxplot(df, 
                 min_n=5)
    
    out_path=os.path.join(results_path, 
                          'debias-oncogv4-across-tasks.pdf')
    
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=[0.1, 0.3,0.5,0.7, 0.9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )
    
    
    ### Fig 1b
    
    reas=pd.read_csv('../results/debias-onevall.csv', index_col=0)
    dts = reas.loc[reas.datasets=='oncog_v4 --> oncog_v4']\
                    .groupby(['Task','Center',  'datasets']
            ).head(1)['Task'].value_counts()
    dts=dts.loc[dts>1]
    
    df=pd.read_csv('../results/debias-onevall.csv', index_col=0)\
            .groupby(['Task','Center', 'datasets']).head(1)
    df=df.loc[df.Task.isin(dts.index)]
    
    plt.figure(figsize=(8,8))
    ax=make_boxplot(df, min_n=5, col='darkblue')
    
    out_path=os.path.join(results_path, 
                          'debias-oncogv4-onevall-across-tasks.pdf')
    
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=[0.1, 0.3,0.5,0.7, 0.9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )
    
    ### Fig 1e
    

    df=pd.read_csv('../results/debias-pairwise.csv', index_col=0)\
                        .groupby(['Task','Center', 'datasets']).head(1)
    
    plt.figure(figsize=(1.2*5,8))
    ax=make_boxplot(df, 
                 x_val='Center')
    
    out_path=os.path.join(results_path, 
                          'debias-oncogv4-across-centers.pdf')
    
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=[0.1, 0.3,0.5,0.7, 0.9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]
    

    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )

    
    ### Fig 1d
    
    df=pd.read_csv('../results/debias-pairwise.csv', index_col=0)\
                        .groupby(['Task','Center', 'datasets']).head(1)
    
    plt.figure(figsize=(1.2*6,8))
    ax=make_boxplot(df, 
                 x_val='datasets')
    
    out_path=os.path.join(results_path, 
                          'debias-across-datasets.pdf')
    
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=[0.1, 0.3,0.5,0.7, 0.9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )

    
    ### Fig 1c

    ord1=[
           'baseline one-v-all min 5' ,
           'debias-m multitask pairwise min 5',
           'baseline pairwise min 5' ,
            'multitask debias one-v-all min 5',
           ]

    pths = glob.glob('../results/*.csv')
    dfs = [ pd.read_csv(pths[i], index_col=0).assign(model=ord1[i])\
            .groupby(['Task','Center', 'datasets']).head(1)
                for i in range(len(pths)) ]
    
    ## filter to one-v-all-tasks with the required min_n
    dfs[0] = dfs[0].loc[dfs[0].Task.isin(dts.index)]
    dfs[-1] = dfs[-1].loc[dfs[-1].Task.isin(dts.index)]
    
    plot_df=pd.concat( dfs )

    plot_df = plot_df.loc[plot_df.datasets=='oncog_v4 --> oncog_v4']


    pal = {'baseline one-v-all min 5':'darkorange',
           'debias-m multitask pairwise min 5':'lightblue',
           'baseline pairwise min 5':'darkorange',
           'multitask debias one-v-all min 5':'darkblue',#'lightblue'
          }



    plt.figure(figsize=(1.2*4,8))
    ax=sns.boxplot(y='auROC', 
                x='model', 
                data=plot_df, 
                order=np.array(ord1)[[0,3,2,1]],
                hue='model', 
                   palette=pal,
                fliersize=0, 
                   dodge=False
                )

    sns.swarmplot(y='auROC', 
                x='model', 
                data=plot_df, 
                order=np.array(ord1)[[0,3,2,1]],
                color='black', 
                s=5
                )
    plt.legend().remove()

    plt.xticks(rotation=90)
    plt.ylim(0,1)
    
    out_path=os.path.join(results_path, 
                          'debias-oncogv4-onevall-across-models.pdf')
    
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=[0.1, 0.3,0.5,0.7, 0.9],
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )

    
if __name__=='__main__':
    for hide_axes in [False, True]:
        main(results_path='../results/plots/', 
             hide_axes=hide_axes)

















