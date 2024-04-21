import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

pd.set_option('display.max_columns', 35)
df = pd.read_csv("diabetes_risk_prediction_dataset.csv")

df = pd.get_dummies(df, columns=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
                                 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
                                 'Itching', 'Irritability', 'delayed healing', 'partial paresis',
                                 'muscle stiffness', 'Alopecia', 'Obesity'])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

X = df.drop('class', axis=1)
y = df['class']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=49)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=49)

classifiers = {
    'Random Forest': RandomForestClassifier(),
    'LightGBM': LGBMClassifier(verbose=-1),
    'GradientBoost': GradientBoostingClassifier(),
    'ExtraTrees': ExtraTreesClassifier()
}

# Train and test each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    test_predictions = clf.predict(X_test)
    print(f'{name} Test Classification Report:')
    print(classification_report(y_test, test_predictions))

    val_predictions = clf.predict(X_val)
    print(f'{name} Validation Classification Report:')
    print(classification_report(y_val, val_predictions))
    auc_roc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
    print(f'{name} AUC-ROC Score: {auc_roc:.4f}')

m = LGBMClassifier(verbose=-1).fit(X, y)
import pickle

pickle.dump(m, open("LGBM.pkl", "wb"))

# a=LGBMClassifier(verbose=-1)
a = pickle.load(open("LGBM.pkl", "rb"))
test_predictions = a.predict(X_test)
X_test.head()

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class DiabetesData(BaseModel):
    Age: int
    Gender_Female: bool
    Gender_Male: bool
    Polyuria_No: bool
    Polyuria_Yes: bool
    Polydipsia_No: bool
    Polydipsia_Yes: bool
    sudden_weight_loss_No: bool
    sudden_weight_loss_Yes: bool
    weakness_No: bool
    weakness_Yes: bool
    Polyphagia_No: bool
    Polyphagia_Yes: bool
    Genital_thrush_No: bool
    Genital_thrush_Yes: bool
    visual_blurring_No: bool
    visual_blurring_Yes: bool
    Itching_No: bool
    Itching_Yes: bool
    Irritability_No: bool
    Irritability_Yes: bool
    delayed_healing_No: bool
    delayed_healing_Yes: bool
    partial_paresis_No: bool
    partial_paresis_Yes: bool
    muscle_stiffness_No: bool
    muscle_stiffness_Yes: bool
    Alopecia_No: bool
    Alopecia_Yes: bool
    Obesity_No: bool
    Obesity_Yes: bool


@app.post("/predictDiabetics")
async def predict_diabetes(data: DiabetesData):
    new_data = data.dict()
    new_data_df = pd.DataFrame([new_data], index=[None])


    print(new_data_df)

    # Use the trained model to predict whether the individual has diabetes or not
    prediction = clf.predict(new_data_df)

    if prediction[0] == 1:
        return ("The individual is predicted to have diabetes.")
    else:
        return ("The individual is predicted not to have diabetes.")
