import os
import time
import gc
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import pickle
import requests
import json
from google.cloud import storage
from google.auth import app_engine

class LoadModel:
    def load():
        #API calls
        request = requests.get('http://api.beerless.be/api/tastingprofiles/averages?access_token=smCLVeBjK79ywuPJFRI599qiu1JFFgKVJrVCq9mtzV0Nus5j5IYB9B8B9uthSTc6')
        x = request.json()
        test = pd.DataFrame(x)
        df_tastingprofiles = test[['beerId', 'malty', 'sweet', 'sour', 'hoppy', 'bitter', 'fruity']]

        # pivot and create tastingprofile matrix
        df_tastingprofile_features = df_tastingprofiles.set_index('beerId')   

        #Configuring Google Cloud storage
        client = storage.Client()
        bucket = client.get_bucket("beerless-scripts-1.appspot.com")
        beerIDPickle = bucket.blob("beerID.pickle")

        # Upload pickle dump
        beerIDPickle.upload_from_string(pickle.dumps(df_tastingprofiles, protocol=pickle.HIGHEST_PROTOCOL))

        #Creating matrix
        mat_tastingprofile_features = csr_matrix(df_tastingprofile_features.values)
        
        #Saving data
        dataPickle = bucket.blob("data.pickle")
        dataPickle.upload_from_string(pickle.dumps(mat_tastingprofile_features, protocol=pickle.HIGHEST_PROTOCOL))

        #creating models
        model = NearestNeighbors()

        #adding parameters to model
        model.set_params(n_neighbors=20, algorithm='brute', metric='cosine',n_jobs=-1)

        # fit
        model.fit(mat_tastingprofile_features)

        #saving model to file
        modelPickle = bucket.blob("model.pickle")
        modelPickle.upload_from_string(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))       

        # clean up
        del df_tastingprofiles, df_tastingprofile_features

if __name__ == "__main__":
    loadModel = LoadModel
    loadModel.load()