import numpy as np
import pandas as pd

def isolate_target(my_df, target = 'salary'):
    if target in list(my_df.columns):
        new_df = my_df.drop(target, axis = 1)
        target_df = my_df[target]
        return (new_df, target_df)
    else:
        return (my_df, np.zeros(len(my_df)))

def identify_interactions(mydf, grouping_columns, resid_column):
    interactions_to_explore = []
    i = 0
    for i in range(len(grouping_columns) - 1):
        for j in range(i+1, len(grouping_columns)):
            interaction_name = str(grouping_columns[i]) + ' and ' + str(grouping_columns[j]) 
            interaction_value = mydf.groupby([grouping_columns[i], grouping_columns[j]])[resid_column].mean()
            interaction_max = np.abs(interaction_value).max()
            if interaction_max > 1:
                print('{} shows interesting interactions, explore further'.format(interaction_name))
                interactions_to_explore.append([grouping_columns[i], grouping_columns[j]])
            else:
                print('{} showed no interesting interactions'.format(interaction_name))
    return interactions_to_explore
    
def get_nlargest_interactions(mydf, interaction, resid_col, n=5):
    interaction_value = mydf.groupby(interaction)[resid_col].mean()
    print('{} largest absolute residual errors for interaction between {} and {}'.format(n, interaction[0], interaction[1]))
    nl = np.abs(interaction_value).nlargest(n)
    print(nl)
    print('\n')
    return nl

def make_interaction_value_bins(mydf, acol, resid_col = 'adv_OLS_resid'):
    comp_col_interact = mydf.groupby(acol)[resid_col].mean()
    comp_col_interact_bin = pd.cut(comp_col_interact, bins = [-99, -2, 2, 99], 
                                   labels = ['significant_negative', 'insignificant', 'significant_positive'])
    comp_col_interact_bin = comp_col_interact_bin.rename('{}_interaction_value'.format(acol))
    return comp_col_interact_bin

def make_interaction_value_col(mydf, acol, interaction_value_bins):
    mydf = pd.merge(mydf, interaction_value_bins[acol], how = 'left', on = acol)
    return mydf
