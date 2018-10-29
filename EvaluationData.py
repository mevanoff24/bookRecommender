from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline


class CreateDataSets:
    def __init__(self, data, popularity_ranks, diversity=False, leave_one_out=False, anti_test=False):
        self.rankings = popularity_ranks
        self.random_state = 100
        
        # Build a full training set for evaluating overall properties
        self.full_train = data.build_full_trainset()
        
        if anti_test:
            self.full_test = self.full_train.build_anti_testset()
        
        # Build a 75/25 train/test split for measuring accuracy
        self.train, self.test = train_test_split(data, test_size=0.25, random_state=self.random_state)
        
        # Build a "leave one out" train/test split for evaluating top-N recommenders
        # Build an anti-test-set for building predictions
        if leave_one_out:
            LOOCV = LeaveOneOut(n_splits=1, random_state=self.random_state)
            for train, test in LOOCV.split(data):
                self.LOOCV_train = train
                self.LOOCV_test = test

            self.LOOCV_anti_test = self.LOOCV_train.build_anti_testset()
        
        # Build interaction matrix for diversity 
        if diversity:
            sim_options = {'name': 'cosine', 'user_based': False}
            self.similarites = KNNBaseline(sim_options=sim_options)
            self.similarites.fit(self.full_train)

#     @property
    def get_full_train(self):
        return self.full_train
    
#     @property
    def get_full_anti_test(self):
        return self.full_test
    
    # needs work
    def get_anti_test_for_user(self, test_row):    
        trainset = self.full_train
        fill = trainset.global_mean
        anti_testset = []
#         u = trainset.to_inner_uid(str(test_row))
        u = trainset.to_inner_uid(test_row)
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if i not in user_items]
        return anti_testset
    
#     @property   
    def get_train(self):
        return self.train
    
#     @property
    def get_test(self):
        return self.test
    
#     @property 
    def get_LOOCV_train(self):
        return self.LOOCV_train
    
#     @property
    def get_LOOCV_test(self):
        return self.LOOCV_test
    
#     @property
    def get_LOOCV_anti_test(self):
        return self.LOOCV_anti_test
    
#     @property
    def get_similarities(self):
        return self.similarites
    
#     @property
    def get_popularity_rankings(self):
        return self.rankings