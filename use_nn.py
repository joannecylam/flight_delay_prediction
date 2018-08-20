# coding: utf-8

from data_loader import PrepareData
from sklearn.model_selection import cross_val_score
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import pandas as pd
from sklearn.neural_network import MLPRegressor

ppd = PrepareData()
df = ppd.load_data()
x, y = ppd.build_train(df)


# In[ ]:


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=33)


# In[ ]:


model = MLPRegressor()
model.fit(train_x, train_y)

print "test x:", test_x.shape

predictions = model.predict(test_x)
print "predictions", predictions.shape
print "test_y:", test_y.shape, test_y.values.shape

#test_y['predictions'] = predictions


ma_r = mean_absolute_error(predictions, test_y.values)
ms_r = mean_squared_error(predictions, test_y.values)
print "mean absolution error:", ma_r
print "mean_squared_error:", ms_r

timestamp = datetime.datetime.now().strftime("%H%M")
test_y.to_csv("results_{}".format(timestamp))

scores = cross_val_score(model, x, y, cv=5)
print "scores:", scores.mean()
