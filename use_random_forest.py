from data_loader import PrepareData
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ppd = PrepareData()
df = ppd.load_data()
x, y = ppd.build_train(df)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=33)

print "Train Random Forest Regressor... "

rf = RandomForestRegressor(n_estimators=90, boostrap=True)
rf.fit(train_x, train_y)
results = rf.predict(test_x)

ma_r = mean_absolute_error(results, test_y.values)
ms_r = mean_squared_error(results, test_y.values)
print "mean absolution error:", ma_r
print "mean_squared_error:", ms_r

## reduce to 3 to save computation time
scores = cross_val_score(rf, x, y, cv=3)
print "score:", scores.mean()
