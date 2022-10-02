# import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle 

# load data 
df = pd.read_csv("/home/onyeogulu/www/node/mlbookcamp-code/AER_credit_card_data.csv")

df.card = (df.card == 'yes').astype(int)

len(df)

# train test split of data 
df_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_train = df_train.card.values
y_test = df_test.card.values

del df_train['card']
del df_test['card']

def predict(df, dv, model):
    cat = df.to_dict(orient='rows')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


with open('card-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

prediction = predict(df_test, dv, model)

card = prediction >= 0.5
print((card == y_test).mean())
print("Done!")