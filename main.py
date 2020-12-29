import math, datetime, json
import pandas as pd
import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_json("gameData.json", orient='records')

df.drop_duplicates(subset=['game_id'], keep='last', inplace=True)

# X = df[['teamA_sval','teamA_off','teamA_def','teamB_sval','teamB_off','teamB_deff']]
# y = df[['teamA_pts','teamB_pts']]

# rand_state = 64
# X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=64)

# model = RandomForestRegressor(n_estimators=200,random_state=64,criterion='mse')
# model.fit(X_train, y_train)
# print(f'Model Score: {model.score(X_test,y_test)}')

# mpr = model.predict([ [0.7853,0.7908,0.8275,0.8300,0.8431,0.8527,] ])[0]
# print(f"Prediction: {mpr[0]} - {mpr[1]}")

X = df[['teamA_sval','teamA_off','teamB_sval','teamB_deff']]
y = df['teamA_pts']
X2 = df[['teamB_sval','teamB_off','teamA_sval','teamA_def']]
y2 = df['teamB_pts']

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=64)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,random_state=64)

model = RandomForestRegressor(n_estimators=1000,random_state=64,criterion='mae')
model.fit(X, y)
# print(f'Model Score: {model.score(X_test,y_test)}')

model2 = RandomForestRegressor(n_estimators=1000,random_state=64,criterion='mae')
model2.fit(X2, y2)
# print(f'Model Score: {model2.score(X2_test,y2_test)}')

predict_num = 1

a_ovr = 0.7433
a_off = 0.8647
a_def = 0.7021

b_ovr = 0.7993
b_off = 0.8530
b_def = 0.7999

if predict_num == 1:
    modelA_predict = [a_ovr, a_off, b_ovr, b_def]
    modelB_predict = [b_ovr, b_off, a_ovr, a_def]
else:
    modelA_predict = [b_ovr, b_off, a_ovr, a_def]
    modelB_predict = [a_ovr, a_off, b_ovr, b_def]

mpr = model.predict([ modelA_predict ])[0]
mpr2 = model2.predict([ modelB_predict ])[0]

print(f"Prediction: {mpr} - {mpr2}")

#region results
# Louisiana - UTSA: []
#     Prediction: [[37.0 - 23.3]]
#     Actual: 31-24

# WKU - GA State: []s
#     Prediction: [[18.7 - 33.0]]
#     Actual: 21-39

# Memphis - FAU: []
#     Prediction: [[27.0 - 12.1]]
#     Actual: 25-10

# Liberty - CCU: [0.7646,0.8400,0.8250,0.8310,0.8622,0.8407]
#     Prediction: [[33.7 - 33.1]]
#     Actual: 37-34

# Buffalo - Marshall: [0.7364,0.8972,0.7603,0.6372,0.7275,0.8739]
#     Prediction1: [[20.3125 - 13.15625]]
#     Prediction2: [[30.145833333333336 - 24.640625]]
#     Actual: 17-10

# Hawaii - Houston: [0.5253,0.7094,0.7296,0.5286,0.8010,0.7458]
#     Prediction: [[30.5 - 20.8]]
#     Actual: 28-14

# UCF - BYU: [0.8002,0.8824,0.8962,0.6406,0.8854,0.6886]
#     Prediction: [[23.015625 - 45.859375]]
#     Actual: 23-49

# App State - North Texas: [0.6656,0.7826,0.8595,0.4853,0.8011,0.5875]
#     Prediction: [[38.375 - 23.7578125]]
#     Actual: 56-28

# Tulane - Nevada: [0.5866,0.7881,0.7718,0.6235,0.8017,0.7958]
#     Prediction: [[28.703125 - 35.71875]]
#     Actual: 27-38

# La Tech - GA Southern: [0.4998,0.6604,0.6684,0.5530,0.7523,0.8141]
#     Prediction: [[9.546875 - 37.265625]]
#     Actual: 3-38

# UNC - Miami: [0.7606,0.9171,0.7752,0.7480,0.8201,0.7947]
#     Prediction: [[51.65625 - 27.109375]]
#     Actual: 62-26

# OK State - Miami: [0.6702,0.7733,0.8116,0.7480,0.8201,0.7947]
#     Prediction1: [[24.078125 - 32.71875]]
#     Prediction2: [[29.015625 - 32.515625]]
#     Prediction3: [[22.390625 - 28.765625]]
#     Prediction4: [[26.1328125 - 31.9453125]]
#     PredictionA: [[26.546875 - 32.6171875]]
#     Actual: TBD

# Texas - Colorado: [0.7382,0.8659,0.7732,0.7043,0.7965,0.7790]
#     Prediction1: [[38.390625 - 32.78125]]
#     Prediction2: [[32.09375 - 32.90625]]
#     Prediction3: [[36.90625 - 31.7421875]]
#     Prediction4: [[36.8828125 - 29.171875]]
#     PredictionA: [[35.2421875 - 32.84375]]
#     Actual: TBD

# Georgia - Cincinnati: [0.8595,0.8576,0.9040,0.8343,0.8577,0.8977]
#     Prediction1: [[28.921875 - 27.65625]]
#     Prediction2: [[27.629167 - 27.1375]]
#     PredictionA: [[28.275521 - 27.396875]]
#     Actual: TBD

# ND - Alabama: [0.8672,0.8784,0.8923,0.9274,0.9920,0.8700]
#     Prediction1: [[29.07589285714286 - 38.54017857142857]]
#     Prediction2: [[30.3671875 - 38.26692708333333]]
#     PredictionA: [[29.72154 - 38.4035528]]
#     Actual: TBD

# Ohio State - Clemson: [0.9049,0.9513,0.8941,0.9229,0.9651,0.9097]
#     Prediction1: [[32.32161458333333 - 38.96875]]
#     Prediction2: [[35.7890625 - 31.121093749999996]]
#     PredictionA: [[34.0553385 - 35.04492187]]
#     Actual: TBD
#endregion