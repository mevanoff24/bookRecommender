import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
from pathlib import Path

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
