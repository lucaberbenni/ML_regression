import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

'''
import train and test df
'''
df = pd.read_csv(
    'data/train-2.csv', 
    index_col=0, 
    parse_dates=True
)
kaagle = pd.read_csv(
    'data/test.csv', 
    index_col=0, 
    parse_dates=True
)

'''
extract month, hour, weekday columns from index
'''
df['month'] = df.index.month
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday

kaagle['month'] = kaagle.index.month
kaagle['hour'] = kaagle.index.hour
kaagle['weekday'] = kaagle.index.weekday

'''
define X, y
'''
X = df.drop(['count', 'registered', 'casual'], axis=1)
y = df['count']

'''
define the transformation pipeline and fit the model
'''
transform = make_pipeline(
    ColumnTransformer(
        transformers=[
            (
                'cat', 
                OneHotEncoder(
                    handle_unknown='ignore', 
                    sparse=False
                ), 
                [
                    'season', 
                    'holiday', 
                    'workingday', 
                    'weather', 
                    'month', 
                    'hour', 
                    'weekday'
                ]
            ), 
            (
                'num', 
                MinMaxScaler(), 
                [
                    'atemp', 
                    'temp', 
                    'humidity', 
                    'windspeed'
                ]
            )
        ]
    )
)

m = make_pipeline(
    transform, 
    RandomForestRegressor()
)
m.fit(X, y)

'''
define hyperparameters to optimize and cross validate
'''
parameters = {
    'randomforestregressor__n_estimators':[100, 120], 
    'randomforestregressor__max_depth':[8, 16]
}
grid_cv = GridSearchCV(
    estimator=m, 
    param_grid=parameters, 
    cv = 5, 
    scoring='r2'
)
grid_cv.fit(X, y)

'''
define best estimator, fit that model and make prediction
'''
m_best = grid_cv.best_estimator_
m_best.fit(X, y)
y_pred = m_best.predict(kaagle)

'''
create a df for kaagle submission and save it as .csv file
'''
output = pd.DataFrame(
    {
        'datetime':kaagle.index, 
        'count':y_pred
    }
)
output.to_csv(
    'data/submission.csv', 
    index=False
)