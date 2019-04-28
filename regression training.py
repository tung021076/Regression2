import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("BITFINEX/BTCJPY", authtoken="FUKfMTsN2saWzhiP6V4u")

#print(df.head())
print(df.tail())

df = df[['High',  'Low',  'Mid',  'Last', 'Bid', 'Ask', 'Volume']]

df['HL_PCT'] = (df['High'] - df['Low']) / df['Last'] * 100.0
df['PCT_change'] = (df['Last'] - df['High']) / df['High'] * 100.0

df = df[['Last', 'HL_PCT', 'PCT_change', 'Volume']]
#print(df.head())
print(df.tail())

forecast_col = 'Last'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVR(gamma='scale')
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

clf = LinearRegression(n_jobs=-1)
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k,gamma='scale')
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)

forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)
