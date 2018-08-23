from sklearn.model_selection import GridSearchCV
from data_loader import PrepareData
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

ppd = PrepareData()
df = ppd.load_data()
x, y = ppd.build_train(df)

params = {
	'hidden_layer_sizes': [(100, 3), (200, 3)],
    'activation': ["tanh", "relu"],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1]
}

rf = GridSearchCV(MLPRegressor(), params, cv=3, verbose=2) 
rf.fit(x, y)
rf.cv_results_['mean_test_score']

print rf.best_score_
print rf.best_params_

