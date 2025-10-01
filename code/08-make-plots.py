import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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

from matplotlib.markers import MarkerStyle
from scipy.stats import combine_pvalues, ttest_1samp
    
def process_pvals(x):
    if x.shape[0]==1:
        return(x.values[0])
    else:
        return(combine_pvalues(x)[1])


def make_scatterlineplot(in_path,
                         out_path, 
                         x_val, 
                         hide_axes, 
                         same_train_test_processing=True, 
                         plot_gihawi=False, 
                         sig_threshold=0.01, 
                         y_val='auROC'
                         ):
    
    df=pd.read_csv(in_path, index_col=0)\
                        .groupby(['Task','Center', 'datasets', x_val]).head(1)
    
    df=df.loc[df.datasets.str.contains('zebra')==False]
    if not plot_gihawi:
        df=df.loc[df.datasets.str.contains('giha')==False]
    else:
        df=df.loc[df.datasets.str.contains('giha')]
        out_path=out_path=out_path[:-4]+'-gihawi.pdf'
    
    if 'read-count-filtering' in in_path:
        df=df.loc[(df.min_read_counts==10)|(df.min_read_counts>=1000)]
        
    if not plot_gihawi:
        if same_train_test_processing:
            df=df.loc[df.train_ds==df.test_ds]
        else:
            df=df.loc[df.train_ds!=df.test_ds]
            out_path=out_path=out_path[:-4]+'cross-study.pdf'
            
    
    if x_val == 'Task':
        df=df.loc[df.datasets=='oncog_v4 --> oncog_v4']
        if 'onevall' in in_path:
            plt.figure(figsize=(8,8))
        else:
            plt.figure(figsize=(12,8))
        
    else:
        plt.figure(figsize=( 1.2 * df[x_val].nunique(), 8 ))
        
        
    col=[ ['cornflowerblue' if 'onevall' in in_path else 'lightblue'][0] 
                                            if 'debias'  in in_path else
           ['salmon' if 'onevall' in in_path else 'darkorange'][0] ][0]
        
        
    sig_dict = ( df.groupby(x_val)['mwu_p_vs_rand']\
                            .apply(process_pvals) < sig_threshold
                       ).to_dict()
    
    six_asterisk = MarkerStyle(r"$\ast$")  # Uses LaTeX's \ast symbol for a true asterisk

    df_tmp=df.sort_values(y_val, ascending=False)
    
    ax = sns.stripplot(x=x_val, 
                       y=y_val, 
                       data=df_tmp, 
                       color=col, 
                       s=15,
                       edgecolor='black', 
                       linewidth=1,
                       )

    # Add vertical lines spanning min to max y-values for each category
    for category in df_tmp[x_val].unique():
        x_pos = list(df_tmp[x_val].unique()).index(category)  # Get x position
        y_min = df_tmp.loc[df_tmp[x_val] == category, y_val].min()
        y_max = df_tmp.loc[df_tmp[x_val] == category, y_val].max()

        plt.vlines(x=x_pos, 
                   ymin=y_min, 
                   ymax=y_max, 
                   color=col,
                   linewidth=5)

        if sig_dict[category]:
            plt.scatter(x_pos,
                        -0.025, 
                        color=col,
                        marker=six_asterisk,
                        edgecolor='black',
                        linewidth=0.5,
                        s=200, 
                        )
    ax.margins(x=0.025)
    plt.xticks(rotation=90)
    ax.set(yticks=np.linspace(0,1,5))
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=np.linspace(0,1,5),
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

    plt.ylim(-0.05,1.05)
    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )
    return(None)


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
                   color=col,
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



def make_plot(in_path,
              out_path, 
              x_val, 
              hide_axes, 
              same_train_test_processing=True, 
              plot_gihawi=False):
    
    df=pd.read_csv(in_path, index_col=0)\
                        .groupby(['Task','Center', 'datasets', x_val]).head(1)
    
    df=df.loc[df.datasets.str.contains('zebra')==False]
    if not plot_gihawi:
        df=df.loc[df.datasets.str.contains('giha')==False]
    else:
        df=df.loc[df.datasets.str.contains('giha')]
        out_path=out_path=out_path[:-4]+'-gihawi.pdf'
    
    if 'read-count-filtering' in in_path:
        df=df.loc[(df.min_read_counts==10)|(df.min_read_counts>=1000)]
        
    if not plot_gihawi:
        if same_train_test_processing:
            df=df.loc[df.train_ds==df.test_ds]
        else:
            df=df.loc[df.train_ds!=df.test_ds]
            out_path=out_path=out_path[:-4]+'cross-study.pdf'
    
    if x_val == 'Task':
        if 'onevall' in in_path:
            plt.figure(figsize=(8,8))
        else:
            plt.figure(figsize=(12,8))
        
    else:
        plt.figure(figsize=( 1.2 * df[x_val].nunique(), 8 ))
        
    
    
        
    ax=make_boxplot(df, 
                    x_val=x_val,
                    min_n=5,
                    col=[ ['cornflowerblue' if 'onevall' in in_path else 'lightblue'][0] 
                                            if 'debias'  in in_path else
                          ['salmon' if 'onevall' in in_path else 'darkorange'][0] ][0]
                    )
    
    ax.set(yticks=np.linspace(0,1,5))
    if hide_axes:
        ax.set_xticks([])
        ax.set(yticks=np.linspace(0,1,5),
               yticklabels=[])
        plt.xlabel(None)
        plt.ylabel(None)
        plt.title(None)
        out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

#     plt.ylim(-0.05,1.05)
    plt.ylim(-0.05, 1.1)
    plt.savefig(out_path, 
                format='pdf',
                bbox_inches='tight', 
                dpi=900
                )
    return(None)


def make_full_method_comparison_plot(
                                    pths = [ '../results/01-baseline-pairwise.csv',
                                             '../results/02-baseline-onevall.csv',
                                             '../results/03-debiasm-pairwise.csv',
                                             '../results/04-debias-onevall.csv', 
                                            '../results/05-debias-pairwise-eval-within-site.csv',
                                             '../results/06.5-negative-control-baseline-pairwise.csv',
                                             '../results/06-negative-control-debias-pairwise.csv'
                                           ],
                                    ord1=[
                                           'baseline pairwise min 5' ,
                                           'baseline one-v-all min 5' ,
                                           'debias-m multitask pairwise min 5',
                                           'multitask debias one-v-all min 5',
                                           'debias within site evals',
                                           'baseline negative control', 
                                           'debias negative control'
                                           ]
                                        ):


    dfs = [ pd.read_csv(pths[i], index_col=0).assign(model=ord1[i])
                for i in range(len(pths)) ]
    
    dfs[-1] = dfs[-1].loc[dfs[-1].shuffle_type=='center_and_site_shuffled_labels']
    dfs[-2] = dfs[-2].loc[dfs[-2].shuffle_type=='center_and_site_shuffled_labels']
    
    plot_df=pd.concat( dfs )

    plot_df = plot_df.loc[plot_df.datasets=='oncog_v4 --> oncog_v4']
    
    
    print('Cross method plot pvals:')
    for tds in plot_df.model.unique():
        qq0 = plot_df.loc[plot_df.model==tds]
        print(tds + ': {:.3e}'.format(ttest_1samp(qq0.auROC.values, 0.5).pvalue))


    pal = {'baseline one-v-all min 5':'salmon',
           'debias-m multitask pairwise min 5':'lightblue',
           'baseline pairwise min 5':'darkorange',
           'multitask debias one-v-all min 5':'cornflowerblue',
           'baseline negative control':'darkorange',
           'debias negative control':'lightblue', 
           'debias within site evals':'lightblue'
           }

    
    for hide_axes in [False, True]:
        ordd = [1,3,0,2]#, 4]
        
        plt.figure(figsize=(1.2*len(ordd),8))
        ax=sns.boxplot(y='auROC', 
                    x='model', 
                    data=plot_df, 
                    order=np.array(ord1)[ordd],
                    hue='model', 
                       palette=pal,
                    fliersize=0, 
                       dodge=False
                    )

        sns.swarmplot(y='auROC', 
                    x='model', 
                    data=plot_df.reset_index(), 
                    order=np.array(ord1)[ordd],
                    color='black', 
                    s=5
                    )
        plt.legend().remove()

        plt.xticks(rotation=90)
        plt.ylim(-0.05,1.1)

        out_path=os.path.join('../results/plots/', 
                              'oncogv4-onevall-and-pairwise-across-models.pdf')
        
        ax.set(yticks=np.linspace(0,1,5))
        if hide_axes:
            ax.set_xticks([])
            ax.set(yticks=ax.get_yticks(),
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
        
        
    
    ### negative control (i.e. randomization) plot
    for hide_axes in [False, True]:
        ordd = [5, 6]
        
        plt.figure(figsize=(1.2*len(ordd),8))
        ax=sns.boxplot(y='auROC', 
                    x='model', 
                    data=plot_df, 
                    order=np.array(ord1)[ordd],
                    hue='model', 
                       palette=pal,
                    fliersize=0, 
                       dodge=False
                    )

        sns.swarmplot(y='auROC', 
                    x='model', 
                    data=plot_df.reset_index(), 
                    order=np.array(ord1)[ordd],
                    color='black', 
                    s=5
                    )
        plt.legend().remove()

        plt.xticks(rotation=90)
        plt.ylim(-0.05,1.05)

        out_path=os.path.join('../results/plots/', 
                              'oncogv4-negative-control-plot.pdf')
        
        ax.set(yticks=np.linspace(0,1,5))
        if hide_axes:
            ax.set_xticks([])
            ax.set(yticks=ax.get_yticks(),
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
    
    
    return(None)

        
def make_read_count_scatterplot():

    
    df_res = pd.read_csv('../results/03-debiasm-pairwise.csv', index_col=0)
    df_res=df_res.loc[df_res.datasets=='oncog_v4 --> oncog_v4']

    df_res['train_upper_n'] = df_res.train_sums.str.split(' - ').str[1].astype(int)
    df_res['train_lower_n'] = df_res.train_sums.str.split(' - ').str[2].astype(int)
    df_res['test_upper_n'] = df_res.test_sums.str.split(' - ').str[0].astype(int)
    df_res['test_lower_n'] = df_res.test_sums.str.split(' - ').str[1].astype(int)

    df_res['train_read_count_1'] = df_res.train_read_counts.str.split(' - ').str[0].astype(float)
    df_res['train_read_count_2'] = df_res.train_read_counts.str.split(' - ').str[1].astype(float)
    df_res['test_read_count_1']  = df_res.test_read_counts.str.split(' - ').str[0].astype(float)
    df_res['test_read_count_2']  = df_res.test_read_counts.str.split(' - ').str[1].astype(float)

    sample_sums=np.sum([
                        df_res['train_upper_n'],
                        df_res['train_lower_n'],
                        df_res['test_upper_n'] ,
                        df_res['test_lower_n'] 
                        ], axis=0
                        )

    sample_sums_overall =np.sum([
                        df_res['train_upper_n'],
                        df_res['train_lower_n'],
                        df_res['test_upper_n'] ,
                        df_res['test_lower_n'] 
                        ], axis=0
                        )

    avg_read_counts = np.sum([df_res['train_read_count_1'],  
                                         df_res['train_read_count_2'],
                                         df_res['test_read_count_1'],
                                         df_res['test_read_count_2']
                                         ], axis=0) / sample_sums

    plt.figure(figsize=(8,8))
    plt.semilogx()
    ax=sns.scatterplot( x = avg_read_counts, 
                     y =  df_res.auROC.values, 
                     s=250, 
                       color='lightblue')
    plt.ylabel('auROC')
    plt.xlabel('Average read count')
    plt.yticks([0, .25, .5, .75, 1])
    ax.set(xticklabels=[], 
           yticklabels=[], 
           xlabel=None, 
           ylabel=None)
    plt.ylim(-0.05, 1.05)
    plt.savefig('../results/plots/03-auroc-vs-read-counts-no-axis.pdf', 
                format='pdf', 
                bbox_inches='tight', 
                dpi=900)
    
    return(None)


def make_additional_boxplots(x_val='Task', 
                             y_val='auROC',
                             sig_threshold=0.01):
    
    ### fig1b
    for in_path in ['../results/03-debiasm-pairwise.csv', 
                    '../results/04-debias-onevall.csv']:
        df=pd.read_csv(in_path, index_col=0)\
                                .groupby(['Task','Center', 'datasets', x_val]).head(1)

        df=df.loc[df.datasets.str.contains('zebra')==False]
        df=df.loc[df.datasets.str.contains('giha')==False]
        df=df.loc[df.datasets=='oncog_v4 --> oncog_v4']
        plt.figure(figsize=(12,8))

        col=[ ['cornflowerblue' if 'onevall' in in_path else 'lightblue'][0] 
                                                if 'debias'  in in_path else
               ['salmon' if 'onevall' in in_path else 'darkorange'][0] ][0]



        sig_dict = ( df.groupby(x_val)['mwu_p_vs_rand']\
                                .apply(process_pvals) < sig_threshold
                           ).to_dict()
        six_asterisk = MarkerStyle(r"$\ast$")  # Uses LaTeX's \ast symbol for a true asterisk

        df_tmp=df.sort_values(y_val, ascending=False)

        for hide_axes in [False, True]:

            out_path = '../results/plots/{}-by-Task.pdf'\
                            .format(in_path.split('/')[-1].split('.')[0])
            
            if '03' in in_path:
                fig, ax = plt.subplots(figsize=(12*1.30385574525,8))
            elif '04' in in_path:
                fig, ax = plt.subplots(figsize=(8,8))

            # Add vertical lines spanning min to max y-values for each category
            for category in df_tmp[x_val].unique():
                x_pos = list(df_tmp[x_val].unique()).index(category)  # Get x position
                y_min = df_tmp.loc[df_tmp[x_val] == category, y_val].min()
                y_max = df_tmp.loc[df_tmp[x_val] == category, y_val].max()

                ax.vlines(x=x_pos, 
                           ymin=y_min, 
                           ymax=y_max, 
                           color=col,
                           linewidth=2.5, 
                           zorder=0
                            )

                if sig_dict[category]:
                    ax.scatter(x_pos,
                                -0.025, 
                                color=col,
                                marker=six_asterisk,
                                edgecolor='black',
                                linewidth=0.5,
                                s=200, 
                                )

            sns.scatterplot(
                x=x_val, 
                y=y_val, 
                data=df_tmp,
                s=100,
                edgecolor='black',
                color=col,
                ax=ax,
                zorder=10, 
                alpha=0.75,
            )

            # Draw the boxplot
            ax=sns.boxplot(
                x=x_val, 
                y=y_val, 
                zorder=15, 
                data=df_tmp,
                color=col,
                showcaps=False,
                boxprops=dict(visible=False),
                whiskerprops=dict(color='lightblue', linewidth=0),
                flierprops=dict(marker=''),  # hide outliers
                medianprops=dict(visible=False, 
                                 linewidth=2.5,
                                 color='black', 
                                 ),
                ax=ax
            )

            categories = df_tmp[x_val].unique()
            category_positions = {cat.get_text(): cat.get_position()[0] for cat in ax.get_xticklabels()}


            # # === Step 3: Plot custom whisker caps ===
            cap_width = 1.25  # how wide the whisker ends are
            v_offset = 0   # vertical offset for top/bottom caps

            for group in categories:
                values = df_tmp[df_tmp[x_val] == group][y_val]
                min_val = values.min()
                max_val = values.max()
                x_center = list(df_tmp[x_val].unique()).index(group)

                # Bvttom cap (min)
                ax.hlines(
                    y=min_val - v_offset,
                    xmin=x_center - cap_width / 2,
                    xmax=x_center + cap_width / 2,
                    color=col,
                    linewidth=2.5, 
                     zorder=25
                )

                ax.hlines(
                    y=max_val + v_offset,
                    xmin=x_center - cap_width / 2,
                    xmax=x_center + cap_width / 2,
                    color=col,
                    linewidth=2.5, 
                    zorder=25
                          )

            ax.margins(x=0)
            plt.xticks(rotation=90)
            ax.set(yticks=np.linspace(0,1,5))
            if hide_axes:
                plt.ylim(-0.05,1.05)
                ax.set_xticks([])
                ax.set(yticks=np.linspace(0,1,5),
                       #ax.get_yticks(),#[0.1, 0.3,0.5,0.7, 0.9],
                       yticklabels=[])
                plt.xlabel(None)
                plt.ylabel(None)
                plt.title(None)
                out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

            plt.ylim(-0.05,1.05)
            plt.savefig(out_path, 
                        format='pdf',
                        bbox_inches='tight', 
                        dpi=900
                        )

def make_crossstudy_boxplots(x_val='datasets', 
                             y_val='auROC', 
                             ):
    css_order = ['old --> salz_24', 
                 'old --> oncog_v4', 
                 'oncog_v4 --> salz_24',
                 'salz_24 --> oncog_v4',
                 'oncog_v4 --> old', 
                 'salz_24 --> old']
    
    def make_boxplot(df,
                     dataset = 'oncog_v4 --> oncog_v4',
                     min_n=5,
                     y_val='auROC',
                     x_val='Task',
                     col='lightblue', 
                     css_order=None ## adding this spec here
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
                       color=col,
                       fliersize=0, 
                       order=css_order
                    )
        sns.swarmplot(x=x_val, 
                      y=y_val, 
                      data=plot_df, 
                      color='black', 
                      s=5,
                      ax=ax, 
                      order=css_order
                      )
        plt.ylim(0,1)
        plt.xticks(rotation=90)
        return(ax)
    
    
    for in_path in ['../results/03-debiasm-pairwise.csv', 
                    '../results/04-debias-onevall.csv']:
    
        df = pd.read_csv(in_path, 
                      index_col=0)
        df = df.loc[(df.train_ds!=df.test_ds)&(df.datasets.str.contains('gihawi')==False)]
        
        df = df.loc[df.datasets.isin(css_order)]

        for hide_axes in [False, True]:
            out_path='../results/plots/{}-by-datasetscross-study.pdf'\
                                .format(in_path.split('/')[-1].split('.')[0])

            plt.figure(figsize=( 1.2 * df[x_val].nunique(), 8 ))
            ax=make_boxplot(df, 
                            x_val=x_val,
                            min_n=5,
                            col=[ ['cornflowerblue' if 'onevall' in in_path else 'lightblue'][0] 
                                                    if 'debias'  in in_path else
                                  ['salmon' if 'onevall' in in_path else 'darkorange'][0] ][0], 
                            css_order=css_order
                            )

            ax.set(yticks=np.linspace(0,1,5))
            if hide_axes:
                ax.set_xticks([])
                ax.set(yticks=np.linspace(0,1,5),
                       yticklabels=[])
                plt.xlabel(None)
                plt.ylabel(None)
                plt.title(None)
                out_path=out_path[:-4] + '-no-axes'+out_path[-4:]
                
            plt.ylim(-0.05,1.1)

            plt.savefig(out_path, 
                        format='pdf',
                        bbox_inches='tight', 
                        dpi=900
                        )        
    
    ghi_order = ['gihawi_23 --> gihawi_23',
                 'old --> gihawi_23',
                 'oncog_v4 --> gihawi_23',
                 'gihawi_23 --> oncog_v4',
                 'gihawi_23 --> old',
                 ]

    for in_path in ['../results/01-baseline-pairwise.csv', 
                    '../results/02-baseline-onevall.csv']:
                    
        
        if 'pairwise' in in_path:
            df1 = pd.read_csv('../results/01-baseline-pairwise.csv', 
                      index_col=0)
            df=df1.loc[df1.datasets.str.contains('gihawi')]

        elif 'onevall' in in_path:
            df1 = pd.read_csv('../results/02-baseline-onevall.csv', 
                              index_col=0)
            df=df1.loc[df1.datasets.str.contains('gihawi')]

        df = df.loc[df.datasets.isin(ghi_order)]
        
        for hide_axes in [False, True]:
            out_path='../results/plots/{}-by-datasets-gihawi.pdf'\
                            .format(in_path.split('/')[-1].split('.')[0])
            

            plt.figure(figsize=( 1.2 * df[x_val].nunique(), 8 ))
            

            ax=make_boxplot(df, 
                            x_val=x_val,
                            min_n=5,
                            col=[ ['cornflowerblue' if 'onevall' in in_path else 'lightblue'][0] 
                                                    if 'debias'  in in_path else
                                  ['salmon' if 'onevall' in in_path else 'darkorange'][0] ][0], 
                            css_order=ghi_order
                            )
            
            ax.set(yticks=np.linspace(0,1,5))
            if hide_axes:
                ax.set_xticks([])
                ax.set(yticks=np.linspace(0,1,5),
                       yticklabels=[])
                plt.xlabel(None)
                plt.ylabel(None)
                plt.title(None)
                out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

            
            plt.ylim(-0.05,1.1)

            plt.savefig(out_path, 
                        format='pdf',
                        bbox_inches='tight', 
                        dpi=900
                        )

            
            
def main():
    
    ## general debias plots
    for in_path in [
                    '../results/03-debiasm-pairwise.csv', 
                    '../results/04-debias-onevall.csv'
                    ]:
    
        for xval in ['Center', 'datasets']:

            out_path = os.path.join('../results/plots/', 
                                    '{}-by-{}.pdf'.format(in_path.split('/')[-1][:-4], xval) )

            [ make_plot(in_path,
                        out_path, 
                        xval, 
                        hide_axes) for hide_axes in [False,True] ]
            
            if xval=='datasets':
                [ make_plot(in_path,
                            out_path, 
                            xval, 
                            hide_axes,
                            same_train_test_processing=False
                            ) for hide_axes in [False,True] ]
    
                


    ## read count filtering plot
    for in_path in ['../results/07-debias-read-count-filtering-pairwise.csv']:
        for xval in ['min_read_counts']:
            out_path = os.path.join('../results/plots/', 
                                    '{}-by-{}.pdf'.format(in_path.split('/')[-1][:-4], xval) )

            [ make_plot(in_path,
                        out_path, 
                        xval, 
                        hide_axes) for hide_axes in [False,True] ]


    ## other debias plots
    for in_path in ['../results/05-debias-pairwise-eval-within-site.csv',
#                     '../results/08-online-debiasm-pairwise.csv'
                   ]:

        for xval in ['datasets']:

            out_path = os.path.join('../results/plots/', 
                                    '{}-by-{}.pdf'.format(in_path.split('/')[-1][:-4], xval) )

            [ make_plot(in_path,
                        out_path, 
                        xval, 
                        hide_axes) for hide_axes in [False,True] ]
    
    ## boxplot combining methods / onevall / pairwise
    make_full_method_comparison_plot()

    ## scatterplot showing read count vs sample size vs performance
    make_read_count_scatterplot()
    
    
    ## zebra-only plot
    df=pd.read_csv('../results/03-debiasm-pairwise.csv', index_col=0)
    df=df.loc[df.datasets.str.contains('zebra')]
    out_path='../results/plots/03-zebra-debiasm-pairwise.pdf'
    for hide_axes in [False, True]:
        plt.figure(figsize=(8,8))
        ax=make_boxplot(df, 
                        x_val='datasets',
                        min_n=5,
                        col='lightblue')
    
        if hide_axes:
            ax.set_xticks([])
            ax.set(yticks=np.linspace(0,1,5),
                   yticklabels=[])
            plt.xlabel(None)
            plt.ylabel(None)
            plt.title(None)
            out_path=out_path[:-4] + '-no-axes'+out_path[-4:]

        plt.ylim(-0.05,1.05)
        plt.savefig(out_path, 
                    format='pdf',
                    bbox_inches='tight', 
                    dpi=900
                    )
    
    make_additional_boxplots()
    make_read_count_scatterplot()
    make_crossstudy_boxplots()
    
    return(None)


if __name__=='__main__':
    main()
