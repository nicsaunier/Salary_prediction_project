#!/usr/bin/env python
# coding: utf-8

# In[85]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Salary Prediction Helper Functions

## Importing libraries:

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

## Data importing and cleaning

def make_valid_training():
    train_features = pd.read_csv('./data/train_features.csv')
    train_salaries = pd.read_csv('./data/train_salaries.csv')
    full_training = train_features.merge(train_salaries, left_on = 'jobId', right_on = 'jobId')
    valid_training = full_training[full_training['salary'] > 0]
    return valid_training

valid_training = make_valid_training()

## Data Exploration

### EDA Helper functions to print information about columns for analysis

def variable_explore(column_name, my_df = valid_training):
    ''' A function that prints information about a column to facilitate analysis.'''
    mycolumn = my_df[column_name]
    
    if mycolumn.dtype == 'int64' or mycolumn.dtype == 'float64':
        print('Column {} is numeric'.format(column_name) + ' Mean {:0.2f}'.format(mycolumn.mean())
             + ' Min {}'.format(mycolumn.min()) + ' Max {}'.format(mycolumn.max()))
    
    else:
        unique_values = list(mycolumn.unique())
        n_values = len(unique_values)
    
        if n_values<10:
            print('Column {} has {} unique values: '.format(column_name, n_values) + str(unique_values))
            return None
        elif n_values <20:
            print('Column {} has between 10 and 20 unique values'.format(column_name)+ str(unique_values))
        elif n_values <100:
            print('Column {} has between 20 and 100 unique values. Further exploration needed'.format(column_name))
        else:
            print('Column {} has more than 100 unique values. Examples include '.format(column_name) + str(unique_values[:2]))
            return None


def go_through_df(my_df):
    ''' A function that loops through each column in the dataframe to get its information'''
    columnNamesList = list(my_df.columns.values)
    for a_column in columnNamesList:
        variable_explore(a_column)
        
### EDA Helper functions to label columns by type, which will be useful later for many things: grouping, plotting, encoding...
        
def get_labels(a_df):
    '''This function goes through all columns in a dataframe and returns a dictionnary with the column names as keys and the
    type of data in those columns as the values.'''
    label_dict = {}
    for acol in a_df.columns:
        if a_df[acol].dtype in ['int64', 'float64', 'int32']:
            label_dict[acol] = 'numeric'
        else:
            nval = a_df[acol].nunique()
            if nval < 10:
                label_dict[acol] = 'small_categorical'
            elif nval < 20:
                label_dict[acol] = 'medium_categorical'
            elif nval <100:
                label_dict[acol] = 'needs_exploration'
            else:
                label_dict[acol] = 'ignore_first'
    return label_dict
            

### Functions to get lists of columns or columns of the dataframe of certain types

def get_colstokeep(my_df, my_type):
    '''Returns a list of columns to keep based on the type specified'''
    coltypes = get_labels(my_df)
    colstokeep = [akey for akey, aval in coltypes.items() if aval in my_type]
    return colstokeep

def get_small_cats(mydf):
    '''returns a list of just the small_categorical columns'''
    small_cats = get_colstokeep(mydf, 'small_categorical')
    return small_cats

def get_numeric_cols(mydf):
    '''returns a list of just the numeric columns'''
    numerics = get_colstokeep(mydf, 'numeric')
    return numerics

def isolate_type(mydf, mytype):
    col_list = get_colstokeep(mydf, my_type = mytype)
    new_df = mydf[col_list]
    return new_df

def isolate_numeric(mydf):
    return isolate_type(mydf, 'numeric')

### EDA Helper functions which makes boxplots of a numeric variable (target by default) by categories and scatterplots
### of numeric variables with the target

def make_boxplot2(mydf, acol, target = 'salary'):
    '''This function makes a boxplot of the target variable grouped by a category in the dataframe'''
    grouped = mydf.groupby(acol)
    df2 = pd.DataFrame({col:vals[target] for col,vals in grouped})
    meds = df2.median()
    meds.sort_values(inplace=True)
    ax = df2[meds.index].boxplot()
    ax.set_title('Boxplot of salaries grouped by {}'.format(acol))
    ax.tick_params(axis = 'x', labelrotation = 45)
    return ax

def make_all_boxplots(mydf):
    '''This function makes boxplots for all of the small categorical variables in the dataframe '''
    small_cats = get_small_cats(mydf)
    n = len(small_cats)
    fig = plt.figure(figsize = (10, 8*n))
    i=n*100 + 11
    for elt in small_cats:
        fig.add_subplot(i)
        make_boxplot2(mydf, elt)
        i += 1
    return None

def make_scatterplots(mydf = valid_training, target = 'salary'):
    numeric_df = isolate_numeric(mydf)
    mycolumns = list(numeric_df.columns)
    mycolumns.remove(target)
    n = len(mycolumns)
    fig, ax = plt.subplots(nrows = 1, ncols = n, figsize = (15, 8))
    for i in range(n):
        ax[i].scatter(numeric_df[mycolumns[i]], numeric_df['salary'], alpha = .01)
        ax[i].set(xlabel=mycolumns[i], ylabel=target)
    return None
            
### EDA Helper functions that split the lowest and highest quintile of the dataframe

target = valid_training['salary']

def get_thresholds(targetvar = target):
    '''Returns the 20th and 80th percentile of the  target value'''
    low_threshold = targetvar.quantile(.2)
    high_threshold = targetvar.quantile(.8)
    return (low_threshold, high_threshold)

def make_low_df(mydf=valid_training, targetvar=target):
    '''Makes a dataframe that contains only the observations in the bottom 20 % of target value'''
    low = get_thresholds(targetvar)[0]
    return mydf[targetvar <= low]

def make_high_df(mydf=valid_training, targetvar=target):
    '''Makes a dataframe that contains only the observations in the top 20% of target value'''
    high = get_thresholds(targetvar)[1]
    return mydf[targetvar > high]

### EDA Helper function that makes side by side histograms of the target variable by category for the full database, 
### lowest quintile and highest quintile

def make_sidebyside_hists(mydf=valid_training):
    '''Makes side by side histograms of the target variable by category for the full database, bottom 20% and top 20%'''
    mylist = [mydf, make_low_df(mydf), make_high_df(mydf)]
    smallcats = get_small_cats(mydf)
    n = len(smallcats)
    fig, ax = plt.subplots(nrows = n, ncols = len(mylist), figsize = (16, n* 6))
    
    i=0
    colors = ['blue', 'red', 'green']
    for row in ax:
        j=0
        for col in row:
            col.hist(mylist[j][smallcats[i]], color = colors[j])
            col.tick_params('x', labelrotation=45)
            j+=1
        i+=1
    fig.tight_layout()
    return None

### The function below is useful because we can then see what the predictions would be if we considered only one categorical
### variable
def make_colgroup_means(mydf):
    ''' Groups variables by category based on a single column and computes the mean for that group.'''
    labelsdict = get_labels(mydf)
    mean_df = mydf
    for acolumn in list(labelsdict.keys()):
        if labelsdict[acolumn]== 'small_categorical' or labelsdict[acolumn] == 'medium_categorical':
            columncatgroups = valid_training.groupby(acolumn)
            colcatmeans = columncatgroups.salary.mean()
            colname = str(acolumn)+'_mean'
            colcatmeans = colcatmeans.rename(colname)
            mean_df = pd.merge(mean_df, colcatmeans, on = acolumn)    
    return mean_df



## Functions to help establish baseline

def get_best_baseline(mydf):
    meanpred_cols = [acol for acol in list(mydf.columns) if '_mean' in acol]
    best_mse = 1e10
    best_predictor = ''
    for a_meanpredcol in meanpred_cols:
        mse = mean_squared_error(y_true = mydf['salary'], y_pred = mydf[a_meanpredcol])
        print('Mean squared error of predictions using {}'.format(a_meanpredcol) + ' : {}'.format(np.round(mse, 0)))
        if mse < best_mse:
            best_mse = np.round(mse, 0)
            best_predictor = a_meanpredcol
    print('\n')
    print('The best baseline prediction takes {} as its predictor and has a mean squared error of {}, which corresponds to a root mean squared error of {}'.format(best_predictor, best_mse, np.round(np.sqrt(best_mse),1)))
    return None

