import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

def main(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['price'])
    y = df['price']

    numeric_features = ['sqft','age','lot','dist_city','beds','baths','garage']
    categorical_features = ['neighborhood']

    preproc = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ])

    pipe = Pipeline([
        ('pre', preproc),
        ('rf', RandomForestRegressor(random_state=0, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    param_grid = {
        'rf__n_estimators':[100,200],
        'rf__max_depth':[10,20,None],
        'rf__min_samples_split':[2,5]
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_absolute_error', verbose=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    os.makedirs('model', exist_ok=True)
    joblib.dump(best, 'model/model.joblib')

    pred = best.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f'best_params: {gs.best_params_}')
    print(f'mae: {mae:.2f}')
    print(f'r2: {r2:.4f}')
    print('model saved to model/model.joblib')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=False, default='data/house_prices.csv')
    args = p.parse_args()
    main(args.data)
