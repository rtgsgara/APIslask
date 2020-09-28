import csv
import os
import pickle
import pandas as pd

from flask import *
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from model import train_model


app = Flask(__name__)
api = Api(app)


if not os.path.isfile('model.pkl'):
    train_model()


model =pickle.load(open('model.pkl', 'rb'))

def feature_engineering(df):
    df['dow'] = df['click_time'].dt.dayofweek.astype('uint16')
    df['doy'] = df['click_time'].dt.dayofyear.astype('uint16')
    df['hour'] = df['click_time'].dt.hour.astype('uint16')
    features_clicks = ['ip', 'app', 'os', 'device']

    for col in features_clicks:
        col_count_dict = dict(df[[col]].groupby(col).size().sort_index())
        df['{}_clicks'.format(col)] = df[col].map(col_count_dict).astype('uint16')

    features_comb_list = [('app', 'device'), ('ip', 'app'), ('app', 'os')]
    for (col_a, col_b) in features_comb_list:
        df1 = df.groupby([col_a, col_b]).size().astype('uint16')
        df1 = pd.DataFrame(df1, columns=['{}_{}_comb_clicks'.format(col_a, col_b)]).reset_index()
        df = df.merge(df1, how='left', on=[col_a, col_b])
    return df

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.files['upload_file']
        posted_data.save(posted_data.filename)
        #print(posted_data)


        '''click_id = posted_data['click_id']
        ip = posted_data['ip']
        appd = posted_data['app']
        device = posted_data['device']
        ost = posted_data['os']
        channel = posted_data['channel']
        click_time = posted_data['click_time']

        tf=pd.DataFrame([{"click_id": click_id ,"ip":ip,"app":appd,"device":device,"os":ost,"channel":channel,"click_time":click_time}])
        tf.to_csv("t1.csv", index=False)
        print(tf)'''
        df1 = pd.read_csv('test_new.csv', parse_dates=['click_time'])
        #df2 = pd.read_csv('t1.csv', parse_dates=['click_time'])
        #print(df1.head())


        test_feature = feature_engineering(df1)
        time = test_feature[['click_time']]
        test_feature.drop(['click_time', 'click_id'], axis=1, inplace=True)

        prediction = model.predict(test_feature)
        test_feature['predicted'] = prediction
        test_feature['click_time'] = time
        test_predicted = test_feature[['ip', 'app', 'device', 'os', 'channel', 'click_time', 'predicted']]

        test_predicted.to_csv('test_final.csv', index=False)
        #print(test_predicted.head())

        file1 = open("usage.txt", "r+")
        f=file1.readline()

        '''if prediction == 0:
            predicted_class = 'attributed'
        elif prediction == 1:
            predicted_class = 'not attributed'''''

        return test_predicted.to_json()




        '''return jsonify({
            'Prediction': predicted_class
        })'''


class Usage(Resource):
    @staticmethod
    def get():

        file1 = open("usage.txt", "r+")
        f=file1.readline()

        return f

class Score(Resource):
    @staticmethod
    def get():

        file1 = open("score.txt", "r+")
        f=file1.read()
        print(f)

        return f


api.add_resource(MakePrediction, '/predict')
api.add_resource(Usage, '/usage')
api.add_resource(Score, '/score')

if __name__ == '__main__':
    app.run(debug=True)

