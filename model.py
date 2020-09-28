from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def train_model():

    df = pd.read_csv('train_filtered.csv', parse_dates=['click_time'])
    # df_sampled=sample(df)
    df_feature = feature_engineering(df)

    df_feature.drop(['click_time', 'attributed_time'], axis=1, inplace=True)
    X = df_feature.drop(['is_attributed'], axis=1)
    y = df_feature['is_attributed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1337)
    classifier = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    start = time.time()
    classifier.fit(X_train, y_train)
    stop = time.time()
    usage = open('usage.txt', 'w')
    usage.write('Training Time:- {} sec'.format(stop - start))
    usage.close()
    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print('Model Training Finished./n/tAccuracy obtained: {}'.format(accuracy))

    score = open('score.txt', 'w')
    score.write('Confusion_matrix:-\n{}\n\nClassification_report:-\n{}\n\nROC_AUC:-\n{}\n\nAccuracy:-\n{}\n\n'.format(
        confusion_matrix(y_test, predictions), classification_report(y_test, predictions),
        roc_auc_score(y_test, predictions), accuracy_score(y_test, predictions)))

    import pickle
    file = open('model.pkl', 'wb')
    pickle.dump(classifier, file)
    file.close()


import time
init_start=time.time()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score
from sklearn.utils import resample


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

#train_model()
