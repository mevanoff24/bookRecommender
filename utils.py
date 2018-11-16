import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from pathlib import Path

from surprise.model_selection import GridSearchCV

DATA_PATH = Path('data/')

import warnings
warnings.filterwarnings('ignore')


def load_feather(filepath, **kwargs):
    '''
    input: (path to feather file)
    read feather file to pandas dataframe
    output: (pandas dataframe)
    '''
    return pd.read_feather(filepath, **kwargs)



def popular_ratings(ratings, user_threshold=100, rating_threshold=100, book_threshold=1):
    '''
    input: (
       ratings: pandas dataframe
       user_threshold: int
       rating_threshold: int 
       book_threshold: int 
    )
    returns a ratings dataframe with the most popular users 
    with a rating count of above user_threshold, users with ratings above rating_threshold
    and books with more than book_threshold ratings.
    output: (pandas dataframe)
    '''
    counts_users = ratings.User_ID.value_counts()
    counts_ratings = ratings.Book_Rating.value_counts()
    sample_ratings = ratings[ratings['User_ID'].isin(counts_users[counts_users >= user_threshold].index)]
    sample_ratings = sample_ratings[ratings['Book_Rating'].isin(counts_ratings[counts_ratings >= rating_threshold].index)]
    isbn_group = sample_ratings.groupby('ISBN', as_index=False)['Book_Rating'].count()
    sample_ratings = sample_ratings[sample_ratings.ISBN.isin(list(isbn_group[isbn_group.Book_Rating > book_threshold].ISBN.values))]
    return sample_ratings




def surprise_gridSearch(model, data, param_grid, metric='rmse', cv=3):
    """
    GridSearchCV to find best hyperparameters
    Args:
        model: (object): surprise model 
        data: (object): surprise trainset 
        param_grid (dict): hyperparameters to tune 
        metric (str): evaluation metric to optimize (rmse or mae)
        cv (int): number of folds in K-Fold CV
    Returns: 
        best_model: (object) surprise model with optimal hyperparameters 
    """
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv)
    grid.fit(data)
    params = grid.best_params[metric]
    best_model = model(**params)
    return best_model


# metrics 
def mse(x, y):
    return np.sqrt(((x-y)**2).mean())

def rmse(x, y): 
    return np.sqrt(mse(x, y))

def mae(x, y): 
    return np.abs((x-y)).mean()

# Training helpers for NN 
def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)

def get_frozen(model_params):
    return (p for p in model_params if not p.requires_grad)

def all_trainable(model_params):
    return all(p.requires_grad for p in model_params)

def all_frozen(model_params):
    return all(not p.requires_grad for p in model_params)

def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False



