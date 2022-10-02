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

# training of model with logistict regression
def train(df, y, C=1.0):
    cat = df.to_dict(orient='rows')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model

dv, model = train(df_train, y_train, C=1.0)

# saving model 
with open('card-model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Model saved!")




