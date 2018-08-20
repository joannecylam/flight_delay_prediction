from data_loader import PrepareData
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import pandas as pd
import xgboost as xgb


ppd = PrepareData()
df = ppd.load_data()
x, y = ppd.build_train(df)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=33)


for x in train_x, test_x, train_y, test_y:
    print x.shape

model = xgb.XGBRegressor()
model.fit(train_x, train_y)

predictions = model.predict(test_x)

test_y['predictions'] = predictions

ma_r = mean_absolute_error(predictions, test_y.values)
ms_r = mean_squared_error(predictions, test_y.values)
print "mean absolution error:", ma_r
print "mean_squared_error:", ms_r
