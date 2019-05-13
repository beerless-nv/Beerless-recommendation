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


app = Flask(__name__)


@app.route("/")
def hello():
    return "Beerless model loader. Unauthorized access not allowed!"


@app.route("/loadModel")
def load():
    headerCron = request.headers.get('X-AppEngine-Cron')
    if headerCron is None:
        return 'Authorized Access Only!'
    else:
        LoadModel.LoadModel.load()
        return "Done"


if __name__ == '__main__':
    app.run(host='0.0.0.0')
    # get args
