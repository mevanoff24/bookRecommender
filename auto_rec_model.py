from surprise import AlgoBase
from surprise import PredictionImpossible
from auto_rec import AutoRec


from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
    
def create_sparse(trainset, N, M, load=True, path='data/Atrain_2.npz'):
    
    if load:
        A = load_npz(path)
        print('sparse dataset loaded')
        return A
    else:     
        A = lil_matrix((N, M))
        for (uid, iid, rating) in trainset.all_ratings():
            i = int(uid)
            j = int(iid)
            A[i, j] = rating
        A = A.tocsr()
        save_npz(path, A)
        print('sparse dataset created and loaded')
        return A

class AutoRecModel(AlgoBase):
    def __init__(self, epochs=10, hidden=100, lr=0.01, batch_size=100, params={}):
        AlgoBase.__init__(self)
        self.epochs = epochs
        self.hidden = hidden
        self.lr = lr
        self.batch_size = batch_size
        
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        N = trainset.n_users
        M = trainset.n_items
        
        trainingMatrix = create_sparse(trainset, N, M, load=False)
        print(trainingMatrix.shape)
            
        autoRec = AutoRec(N, M, hidden=self.hidden, lr=self.lr, 
                          batch_size=self.batch_size, epochs=self.epochs)
        autoRec.train(trainingMatrix)

        self.predictedRatings = autoRec.model.predict(trainingMatrix)    
    
    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        rating = self.predictedRatings[u, i]
        
        if (rating < 0.001):
            raise PredictionImpossible('No valid prediction exists.')
            
        return rating