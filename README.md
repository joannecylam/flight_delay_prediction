# Flight Delay Prediction

## Preprocessing:
1. Categorical variables are converted to binary variables.
2. A subset (40%) of the training data (named "flight_delay_data_subset.csv") is used in model testing and hyperparameter tunning.
3. Constant variable "Departure" is dropped to reduce training cost.

## Results Evaluation
Total three different models are applied. <br />
Readers may re-produce the model by running the corresponding script. <br />
### Best Result using Random Forest Regressor<br />
![alt text](https://github.com/joannecylam/flight_delay_prediction/blob/master/prediction_vs_true_value_plot.png)

### Environment Requirements:
1. python 2.7
2. sklearn
3. pandas
4. matplotlib

### Models:
1: Random Forest
 - use_random_forest.ipynb<br />
2: Xgboost
 - use_xgboost_regressor.ipynb<br />
3: MLP Regressor
 - use_nn.ipynb

### Visualizations:
There is a scattered plot of the best prediction results against actual value. <br />
The code can be found in visualization.ipynb
