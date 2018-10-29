import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import save_npz, load_npz

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dropout, Dense
from keras.regularizers import l2
from keras.optimizers import SGD


class AutoRec(object):
    def __init__(self, N, M, epochs=5, hidden=500, lr=0.1, batch_size=128, reg=0.01):
        self.N = N
        self.M = M
        self.epochs = epochs
        self.hidden = hidden
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
            
    def train(self, X_train):
#         X_test = load_npz("data/Atest.npz")
        mask = (X_train > 0) * 1.0
#         mask_test = (X_test > 0) * 1.0

        # make copies since we will shuffle
        X_train_copy = X_train.copy()
        mask_copy = mask.copy()
#         X_test_copy = X_test.copy()
#         mask_test_copy = mask_test.copy()
        
        self.mu = X_train.sum() / mask.sum()
        
        self.build()
        
        self.output = self.model.fit_generator(
              self.generator(X_train, mask),
#               validation_data=self.test_generator(X_train_copy, mask_copy, X_test_copy, mask_test_copy),
              epochs = self.epochs,
              steps_per_epoch = X_train.shape[0] // self.batch_size + 1,
#               validation_steps = X_test.shape[0] // self.batch_size + 1,
        )
        
        
    
    def generator(self, X, mask):
        while True:
            X, mask = shuffle(X, mask)
            for i in range(X.shape[0] // self.batch_size + 1):
                upper = min((i + 1) * self.batch_size, X.shape[0])
                a = X[i * self.batch_size:upper].toarray()
                m = mask[i * self.batch_size:upper].toarray()
                a = a - self.mu * m # must keep zeros at zero!
                # m2 = (np.random.random(a.shape) > 0.5)
                # noisy = a * m2
                noisy = a # no noise
                yield noisy, a


    def test_generator(self, X, mask, X_test, mask_test):
      # assumes A and A_test are in corresponding order
      # both of size N x M
        while True:
            for i in range(X.shape[0] // self.batch_size + 1):
                upper = min((i + 1) * self.batch_size, X.shape[0])
                a = X[i * self.batch_size:upper]#.toarray()
                m = mask[i * self.batch_size:upper]#.toarray()
                at = X_test[i * self.batch_size:upper]#.toarray()
                mt = mask_test[i * self.batch_size:upper]#.toarray()
                a = a - self.mu * m
                at = at - self.mu * mt
                yield a, at
                
    def mse_mask_loss(self, y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
        diff = y_pred - y_true
        sqdiff = K.square(diff) * mask
        sse = K.sum(K.sum(sqdiff))
        n = K.sum(K.sum(mask))
        return sse / n

    def rmse_mask_loss(self, y_true, y_pred):
        mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
        diff = y_pred - y_true
        sqdiff = K.square(diff) * mask
        sse = K.sum(K.sum(sqdiff))
        n = K.sum(K.sum(mask))
        mse = sse / n
        return K.sqrt(mse)
    
    def build(self):       
        # build the model - just a 1 hidden layer autoencoder
        input_ = Input(shape=(self.M,))
        X = Dropout(0.2)(input_)
        X = Dense(self.hidden, activation='tanh', kernel_regularizer=l2(self.reg))(X)
        X = Dense(self.M, kernel_regularizer=l2(self.reg))(X)
        
        
        self.model = Model(input_, X)
        self.model.compile(loss=self.mse_mask_loss,
                      optimizer=SGD(lr=0.08, momentum=0.9),
                      metrics=[self.rmse_mask_loss],
        )

    def get_recommendations(self, inputUser):
        pred = self.model.predict([inputUser]) + self.mu
        return pred 
