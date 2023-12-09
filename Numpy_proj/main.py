import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# A popular strategy in direct marketing is the telemarketing phonecalls; even if this kind of intervention is a low-cost alternative, the sucess of its implementation relies in the proper targeting of potential clients.
#
# The following dataset provide information on the the success of telemarketing calls for selling a particular bank product. The dataset contains different features types. Namely:
#
# Client information:

# age: Age of the potential client
# job: admin., blue- collar, entrepreneur, housemaid...
# marital_status: married, single, unknown
# education:basic.4y,basic.6y,basic.9y,high.school,illiterate,...
# Client-Bank Relation
#
# default: The client has credit in default: no,yes,unknown
# housing: The client has a housing loan contract: no,yes,unknown
# loan: The client has a personal loan: no,yes,unknown
# Campain
#
# contact: Communication type (cellular,telephone)
# month: Last month contacted (jan, feb ,..., dec)
# day_of_week: Last contact day : (mon, tue,..., fri)
# duration: Last contact duration (in seconds)
# campain: Number of contacts performed during this campaign
# pdays: of days that passed by after last contact (999 if the client was not previously contacted)
# previous: Number of contacts performed before this campaign
# poutcome: Outcome of the previous marketing campaign: failure,nonexistent,success
# Economic indicators
#
# emp.var.rate: numerical Employment variation rate in the last quarter
# cons.price.idx: numerical Consumer price index in the last month
# cons.conf.idx: numerical Monthly consumer confidence index
# euribor3m: numerical Dayly Euro Interbank Offered Rate
# nr.employed: numerical Number of employeed in the last quarter


def train_model():
    # Load the training data
    file_path_training = 'datasets/telemarketing.csv'
    training_data = pd.read_csv(file_path_training)

    # Preparing the data
    # Identifying categorical and numerical columns
    categorical_cols = training_data.select_dtypes(include=['object']).columns
    numerical_cols = training_data.select_dtypes(include=['int64', 'float64']).columns.drop('target')

    # Creating transformers for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundling preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Splitting data into training and testing sets
    X = training_data.drop('target', axis=1)
    y = training_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Creating a logistic regression pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(max_iter=1000))])  # Increased max_iter for convergence

    # Training the model
    model.fit(X_train, y_train)

    # Predicting and evaluating the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", report)
