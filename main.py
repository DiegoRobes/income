import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df_data = pd.read_csv('NLSY97_subset.csv')[['EARNINGS', 'S', 'EXP']]

df_data.duplicated().values.sum()
df_data.drop_duplicates(inplace=True)

plt.figure(figsize=(8, 4), dpi=200)
with sns.axes_style("darkgrid"):
    sns.pairplot(df_data,
                 kind='reg',
                 plot_kws={
                     'line_kws': {'color': 'black', 'lw': 1},
                     'scatter_kws': {'color': '#2f4b7c', 'alpha': 0.4, 's': 4}
                 })
plt.show()

for column in df_data.columns.values:
    plt.figure(figsize=(12, 4), dpi=200)
    with sns.axes_style("darkgrid"):
        sns.displot(data=df_data, x=column, aspect=2, kde=True, color='#2196f3', bins=50)
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(df_data.drop(['EARNINGS', 'EXP'], axis=1), df_data.EARNINGS, test_size=0.2, random_state=1)

# % of training set
train_pct = 100*len(X_train)/len(df_data.drop('EARNINGS' , axis=1))
print(f'Training data is {train_pct:.3}% of the total data.')

# % of test data set
test_pct = 100*X_test.shape[0]/df_data.drop('EARNINGS' , axis=1).shape[0]
print(f'Test data makes up the remaining {test_pct:0.3}%.')

regr = LinearRegression().fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)
print(f'Training data r-squared: {rsquared:.2}')

regr_coef = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Coefficient'])

expects = regr_coef.loc['S'].values[0]
print(f'The increment of earnings per hour for having an extra year of schooling is ${expects:.5}')

# Using X_train to predict
predict_train = regr.predict(X_train)
residuals = (y_train - predict_train)

# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_train, y=predict_train, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.figure(dpi=100)
plt.scatter(x=predict_train, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

mae_train = mean_absolute_error(y_train, predict_train)
print(f"The absolute mean for the residuals of training data is {mae_train:.5}")

# Residual Distribution Chart
resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()

# Using X_test to predict
predict_test = regr.predict(X_test)
residuals = (y_test - predict_test)

# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_test, y=predict_test, c='indigo', alpha=0.6)
plt.plot(y_test, y_test, color='cyan')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.figure(dpi=100)
plt.scatter(x=predict_test, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

mae_test = mean_absolute_error(y_test, predict_test)
print(f"The absolute mean for the residuals of training data is {mae_test:.5}")

# Residual Distribution Chart
resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()

# ------- multivariable regression ------ #
X_train, X_test, y_train, y_test = train_test_split(df_data.drop(['EARNINGS'], axis=1), df_data.EARNINGS, test_size=0.2, random_state=1)

regr = LinearRegression().fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)
print(f'Training data r-squared: {rsquared:.2}')

regr_coef = pd.DataFrame(data=regr.coef_, index=X_train.columns, columns=['Coefficient'])

# Using X_train to predict
predict_train = regr.predict(X_train)
residuals = (y_train - predict_train)

# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_train, y=predict_train, c='indigo', alpha=0.6)
plt.plot(y_train, y_train, color='cyan')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.figure(dpi=100)
plt.scatter(x=predict_train, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

mae_train = mean_absolute_error(y_train, predict_train)
print(f"The absolute mean for the residuals of training data is {mae_train:.5}")

# Residual Distribution Chart
resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()


# Using X_test to predict
predict_test = regr.predict(X_test)
residuals = (y_test - predict_test)

# Original Regression of Actual vs. Predicted Prices
plt.figure(dpi=100)
plt.scatter(x=y_test, y=predict_test, c='indigo', alpha=0.6)
plt.plot(y_test, y_test, color='cyan')
plt.title(f'Actual vs Predicted Prices: $y _i$ vs $\hat y_i$', fontsize=17)
plt.xlabel('Actual prices 000s $y _i$', fontsize=14)
plt.ylabel('Prediced prices 000s $\hat y _i$', fontsize=14)
plt.show()

# Residuals vs Predicted values
plt.figure(dpi=100)
plt.scatter(x=predict_test, y=residuals, c='indigo', alpha=0.6)
plt.title('Residuals vs Predicted Values', fontsize=17)
plt.xlabel('Predicted Prices $\hat y _i$', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.show()

mae_test = mean_absolute_error(y_test, predict_test)
print(f"The absolute mean for the residuals of training data is {mae_test:.5}")

# Residual Distribution Chart
resid_mean = round(residuals.mean(), 2)
resid_skew = round(residuals.skew(), 2)

sns.displot(residuals, kde=True, color='indigo')
plt.title(f'Residuals Skew ({resid_skew}) Mean ({resid_mean})')
plt.show()

schooling_year = 16
work_experience = 5

new_features = pd.DataFrame(columns=['S', 'EXP'], data=[['16', '5']])
new_prediction = regr.predict(new_features)[0]
print(f"A person with schooling of {schooling_year} years & work experience of {work_experience} "
      f"years is predicted to have earnings of ${new_prediction:.2}")

