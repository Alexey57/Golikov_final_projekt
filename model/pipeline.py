import pandas as pd
import dill

from datetime import datetime, date, time
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate

from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import pickle
import json


def main():

    def hits_preparation():
        df_hits = pd.read_csv('C:/Users/папа/homework_final_project/data_final_projekt/ga_hits.csv').drop([
            'hit_number', 'hit_type', 'hit_page_path', 'event_category', 'event_value', 'hit_time', 'hit_referer',
            'event_label', 'hit_date'], axis=1)
        df_hits = df_hits.drop_duplicates()
        target = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                  'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                  'sub_submit_success', 'sub_car_request_submit_click']
        # df_hits.loc[:, 'target_action'] = df_hits['event_action'].apply(lambda x: 'yes' if x in target else 'no')
        df_hits.loc[:, 'target_action'] = df_hits['event_action'].apply(lambda x: 1 if x in target else 0)
        df_hits = df_hits.drop(['event_action'], axis=1)
        df_hits = df_hits.drop_duplicates()

        return df_hits

    def sessions_preparation():
        df_sessions = pd.read_csv('C:/Users/папа/homework_final_project/data_final_projekt/ga_sessions.csv',
            low_memory=False).drop(['device_model', 'device_brand', 'utm_keyword', 'device_os', 'utm_campaign',
            'utm_adcontent', 'visit_number', 'device_screen_resolution', 'client_id', 'device_browser',
            'visit_time'], axis=1)
        df_sessions = df_sessions[df_sessions['utm_source'].isna() == False]
        df_sessions = df_sessions[df_sessions.geo_country == 'Russia']
        df_sessions = df_sessions.drop(['geo_country'], axis=1)
        df_sessions = df_sessions.drop_duplicates()

        return df_sessions

    def association_df():
        df_hits = hits_preparation()
        df_sessions = sessions_preparation()
        df_hits_sessions = df_sessions.merge(df_hits, on=["session_id"])
        drop_date = ['2021-05-24', '2021-05-25', '2021-12-21']
        df_hits_sessions = df_hits_sessions[df_hits_sessions.visit_date.isin(drop_date) == False]
        df_hits_sessions = df_hits_sessions[df_hits_sessions.geo_city != '(not set)']

        return df_hits_sessions

    df = association_df()
    df = df.drop(['session_id', 'visit_date'], axis=1)
    df = df.drop_duplicates()

    X = df.drop(['target_action'], axis=1)
    y = df.target_action

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))

    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC(kernel='linear')
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    with open('C:/Users/папа/PycharmProjects/Golikov_final_projekt/model/data/model_hits_sessions_binar.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Event action prediction model',
                'author': 'Golikov Aleksey',
                'version': 1,
                'date': datetime.now(),
                'type': type(best_pipe.named_steps["classifier"]).__name__,
                'accuracy': best_score
            }
        }, file)


if __name__ == '__main__':
    main()