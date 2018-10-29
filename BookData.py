from collections import defaultdict
from surprise import Dataset, SVD, Reader, evaluate, print_perf
from surprise.dataset import DatasetAutoFolds
from surprise.model_selection import train_test_split, LeaveOneOut
from surprise import KNNBaseline


class BookDataSet(DatasetAutoFolds):
    def __init__(self, ratings, books, users):
        self.ratings = ratings
        self.books = books
        self.users = users
        self.reader = reader = Reader(line_format='user item rating', rating_scale=(1, 10))
        self.raw_ratings = [(uid, iid, r, None) for (uid, iid, r) in 
                            zip(ratings['User_ID'], ratings['ISBN'], ratings['Book_Rating'])]
        
    def load_ratings_dataset(self, remove_outliers=False):        
        if remove_outliers:
            outlier_threshold = 3
            user_group = self.ratings.groupby('User_ID', as_index=False).agg({'Book_Rating': 'count'})
            user_group['outlier'] = (abs(user_group.Book_Rating - user_group.Book_Rating.mean()) > user_group.Book_Rating.std() * outlier_threshold)
            user_group.drop('Book_Rating', axis=1, inplace=True)
            self.ratings = self.ratings.merge(user_group, on='User_ID', how='left')
            self.ratings = self.ratings[self.ratings.outlier == False]
            self.ratings.drop('outlier', axis=1, inplace=True)
            ratings_dataset = Dataset.load_from_df(self.ratings, reader=self.reader)
        else:
            ratings_dataset = Dataset.load_from_df(self.ratings, reader=self.reader)
        return ratings_dataset
        
    def get_user_ratings(self, user_id):
        user = self.ratings[self.ratings['User_ID'] == user_id].values
        return [(info[1], info[2]) for info in user]
        
    def get_popularity_ranks(self):
        return defaultdict(int, self.ratings.groupby('ISBN')['Book_Rating'].count().sort_values().to_dict())
    
    def get_year(self, fill_na=False):
        years = self.books['Year_Of_Publication']
        if fill_na: 
            years = self.books['Year_Of_Publication'].fillna(self.books['Year_Of_Publication'].median()).astype(int)
        return defaultdict(int, zip(self.books['ISBN'], years))
    
    def _get_book(self, attribute, col='ISBN'):
        return self.books[self.books[col] == attribute]
    
    def _get_book_attribute(self, isbn, attribute):
        book = self._get_book(isbn)
        return book[attribute].values[0] if not book.empty else ''
        
    def get_book_name(self, isbn):
        return self._get_book_attribute(isbn, 'Book_Title')
    
    def get_book_year(self, isbn):
        return self._get_book_attribute(isbn, 'Year_Of_Publication')
    
    def get_book_author(self, isbn):
        return self._get_book_attribute(isbn, 'Book_Author')
    
    def get_book_publisher(self, isbn):
        return self._get_book_attribute(isbn, 'Publisher')
    
    def get_book_id(self, book_name):
        book = self._get_book(book_name, col='Book_Title')
        return book['ISBN'].values[0] if not book.empty else ''
