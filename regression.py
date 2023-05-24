import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2, 
                                                    random_state = 42)
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

X_train_t = transform.fit_transform(X_train)
X_test_t = transform.transform(X_test)
m = RandomForestRegressor()

m.fit(X_train_t, y_train)


'''
define hyperparameters to optimize and cross validate
'''
parameters = {
    'n_estimators':[100, 120], 
    'max_depth':[8, 16]
}
grid_cv = GridSearchCV(
    estimator=m, 
    param_grid=parameters, 
    cv = 5, 
    scoring='r2'
)
grid_cv.fit(X_train_t, y_train)

'''
define best estimator, fit that model and make prediction
'''
m_best = grid_cv.best_estimator_
m_best.fit(X_train_t, y_train)
m_pred = m_best.predict(X_test_t)

print(r2_score(y_test, m_pred))

kaagle_t = transform.transform(kaagle)
y_pred = m_best.predict(kaagle_t)

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