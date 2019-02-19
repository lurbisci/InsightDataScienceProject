from flask import Flask, render_template, request

#Import the necessary Python packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential, Model, load_model
import json



def read_data():
    # Read in the data
    whisky_clean = pd.read_csv("./whiskyapp/whisky_clean2.csv")

    # Drop columns
    whisky_clean = whisky_clean.drop(columns=['Rating_category', 'Rating_category_quantile'], axis=1)

    # Encode labels with value between 0 and n_classes-1
    le = preprocessing.LabelEncoder()

    whisky_clean.loc[:, 'Whisky'] = le.fit_transform(whisky_clean['Whisky'].astype(str))
    whisky_clean.loc[:, 'Class'] = le.fit_transform(whisky_clean['Class'].astype(str))
    whisky_clean.loc[:, 'Cluster'] = le.fit_transform(whisky_clean['Cluster'].astype(str))
    whisky_clean.loc[:, 'Type'] = le.fit_transform(whisky_clean['Type'].astype(str))
    whisky_clean.loc[:, 'Reddit_Username'] = le.fit_transform(whisky_clean['Reddit_Username'].astype(str))
    whisky_clean.loc[:, 'Region_cleaned'] = le.fit_transform(whisky_clean['Region_cleaned'].astype(str))
    return whisky_clean

with open("./whiskyapp/dictionary_whisky.json") as f:
    whisky_dict = json.load(f)

model = load_model('./whiskyapp/whisky_cf_nonneg.h5')
dat = read_data()

whi = model.get_layer("NonNegWhisky-Embedding")
whisky_weights = whi.get_weights()[0]
lens = np.linalg.norm(whisky_weights, axis=1)
normalized = (whisky_weights.T / lens).T

def similarity_output(whisky_type):
    search_num = int(whisky_type) #accessing the zeroth element
    dists = np.dot(normalized, normalized[search_num])
    return dists

def recommendation_output(whisky_type, cost_index, reddit_index):
    cost_index = int(cost_index) #accessing the zeroth element
    reddit_index = float(reddit_index) #accessing the zeroth element

    dists = similarity_output(whisky_type)
    whisky_group = dat.groupby(['Whisky'], as_index=False).mean()
    whisky_dict_reverse = {v: k for (k, v) in whisky_dict.items()}
    whisky_group['Whisky_name'] = whisky_group['Whisky'].apply(lambda x: whisky_dict_reverse[x])
    whisky_group['Dot'] = dists
    whisky_group.sort_values(by=['Dot'], inplace=True, ascending=False)
    whisky_group['Reddit_Review'] = whisky_group['Reddit_Review'].astype(int)
    whisky_out = whisky_group.loc[(whisky_group['Cost_num'] == cost_index) & (whisky_group['Reddit_Review'] > reddit_index),
                     ['Whisky_name', 'Cost_num', 'Reddit_Review', 'Meta_Critic', 'Cluster',
                      'Class', 'Type']]
    return whisky_out


# App part
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/input')
def whisky_input():
    return render_template("input.html")

@app.route('/output')
def whisky_output():
    whisky_type = request.args.get("whisky_type")
    cost_index = request.args.get("cost_index")
    reddit_index = request.args.get("reddit_index")
    whisky_results = recommendation_output(whisky_type, cost_index, reddit_index)
    #print(len(whisky_results))
    #if len(whisky_results) == 0:
    #    return render_template("output.html")
    return render_template("output.html", whisky_names=whisky_results['Whisky_name'].values,
                           redditreview=whisky_results['Reddit_Review'].values)

@app.route('/about')
def whisky_about():
    return render_template("about.html")

