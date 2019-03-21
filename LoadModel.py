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

class LoadModel:
    def load():
        #API calls
        request = requests.get('http://api.beerless.be/api/tastingprofiles/averages?access_token=smCLVeBjK79ywuPJFRI599qiu1JFFgKVJrVCq9mtzV0Nus5j5IYB9B8B9uthSTc6')
        x = request.json()
        test = pd.DataFrame(x)
        #print(test.columns.values)
        df_tastingprofiles = test[['beerId', 'malty', 'sweet', 'sour', 'hoppy', 'bitter', 'fruity']]
        df_tastingprofile_mean = df_tastingprofiles

        # pivot and create tastingprofile matrix
        df_tastingprofile_features = df_tastingprofiles.set_index('beerId')   
        #print(df_tastingprofile_features) 

        #Configuring Google Cloud storage
        client = storage.Client()
        bucket = client.get_bucket("beerless-scripts-1")
        beerIDPickle = bucket.blob("beerIDPickle")

        beerIDPickle.upload_from_string(pickle.dumps(df_tastingprofiles, protocol=pickle.HIGHEST_PROTOCOL))

        #Creating matrix
        mat_tastingprofile_features = csr_matrix(df_tastingprofile_features.values)
        
        #Saving data
        dataPickle = bucket.blob("dataPickle")
        dataPickle.upload_from_string(pickle.dumps(mat_tastingprofile_features, protocol=pickle.HIGHEST_PROTOCOL))

        #creating models
        model = NearestNeighbors()

        #adding parameters to model
        model.set_params(n_neighbors=20, algorithm='brute', metric='cosine',n_jobs=-1)


        # fit
        model.fit(mat_tastingprofile_features)

        #saving model to file
        modelPickle = bucket.blob("modelPickle")
        modelPickle.upload_from_string(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            

        # clean up
        del df_tastingprofiles, df_tastingprofile_features

if __name__ == "__main__":
    loadModel = LoadModel
    loadModel.load()