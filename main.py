import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import pickle
import sys
import LoadModel
from google.cloud import storage
from flask import Flask, request
from google.auth import app_engine
import requests


class KnnRecommender:
    def _prep_data(self):
        # Get model, hashmap and data
        # Connect to GCP bucket
        client = storage.Client()
        bucket = client.get_bucket("beerless-scripts-1.appspot.com")

        # Model
        modelPickle = bucket.blob("model.pickle")
        model = pickle.loads(modelPickle.download_as_string())

        # Data
        dataPickle = bucket.blob("data.pickle")
        data = pickle.loads(dataPickle.download_as_string())

        # Tastingprofiles
        beerIdPickle = bucket.blob("beerID.pickle")
        df_tastingprofiles = pickle.loads(beerIdPickle.download_as_string())

        return model, data, df_tastingprofiles

    def _inference(self, model, data, idx, n_recommendations):
        """
        return top n similar beer recommendations based on user's input movie
        Parameters
        ----------
        model: sklearn model, knn model
        data: beer-tastingprofile matrix
        hashmap: dict, map beer name to index of the beer in data
        fav_beer: str, name of user input beer
        n_recommendations: int, top n recommendations
        Return
        ------
        list of top n similar beer recommendations
        """
        # Removing duplicate recommendation
        n_recommendations = (n_recommendations * 2) + 1

        # inference
        distances, indices = model.kneighbors(
            data[idx],
            n_neighbors=n_recommendations +1)

        # get list of raw idx of recommendations
        raw_recommends = \
            sorted(
                list(
                    zip(
                        indices.squeeze().tolist(),
                        distances.squeeze().tolist()
                    )
                ),
                key=lambda x: x[1]
            )[:0:-1]

        return raw_recommends

    def make_recommendations(self, idx, n_recommendations):
        """
        make top n beer recommendations
        Parameters
        ----------
        fav_beer: str, name of user input beer
        n_recommendations: int, top n recommendations
        """

        # get data
        model, data, df_tastingprofiles = self._prep_data()

        if(df_tastingprofiles[df_tastingprofiles['beerId'] == idx].index.values > -1):

            # get index of beerId
            beerindex = int(
                df_tastingprofiles[df_tastingprofiles['beerId'] == idx].index.values)

            # get recommendations
            raw_recommends = self._inference(
                model, data, beerindex, n_recommendations)

            # Sorting Raw Recommendations
            def sortSecond(val):
                return val[1]

            raw_recommends.sort(key=sortSecond)

            # Create object to return
            beerAPI = []

            # Run through every recommend
            for beer in raw_recommends:
                beerID = int(df_tastingprofiles.iloc[beer[0]].beerId)

                #  Check if beer in recommendations = rec beer
                if beer[0] != beerindex:
                    beerAPI.append({
                        'beerId': beerID,
                        'distance': beer[1]
                    })

            # Return object
            beerAPIresult = pd.DataFrame(beerAPI)
            return beerAPIresult.to_json(orient='records')
        else:
            test = []
            response = pd.DataFrame(test)
            return response.to_json(orient='records')


app = Flask(__name__)


@app.route("/")
def hello():
    return "Beerless API"


@app.route("/itemBasedRecommendation", methods=['GET'])
def getRecommendation():
    # Get ARgs
    beerId = int(request.args['beerId'])
    amount = int(request.args['amount'])

    # Check if beerId exists
    # Call to API to check beer
    baseUri = "https://api.beerless.be/api/beers/{}".format(beerId)
    result = requests.get(baseUri)
    if result.status_code == 200:
        # initial recommender system
        recommender = KnnRecommender()
        # make recommendations
        return recommender.make_recommendations(beerId, amount)
    else:
        test = []
        response = pd.DataFrame(test)
        return "YOLO MCSWAGGINS"


@app.route("/loadModel")
def load():
    headerCron = request.headers.get('X-AppEngine-Cron')
    if headerCron is None:
        return 'Authorized Access Only!'
    else:
        LoadModel.LoadModel.load()
        return "Done"


if __name__ == '__main__':
    app.run(debug=True)
    # get args
