from data_loader import PrepareData
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import xgboost as xgb

ppd = PrepareData()
ppd = load_data()
x, y = ppd.build_train(df)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=33)


ma_r = mean_absolute_error(results, test_y.values) 
ms_r = mean_squared_error(results, test_y.values) 
print "mean absolution error:", ma_r
print "mean_squared_error:", ms_r

model = xgb.XGBRegressor()
model.fit(train_x,train_y)
predictions = model.predict(test_x)
test_y['predictions'] = predictions

timestamp = datetime.datetime.now().strftime("%H%M")
test_y.to_csv("results_{}".format(timestamp))
pickle.dump(model, open("model_{}".format(timestamp), 'rb'))