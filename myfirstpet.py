import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, accuracy_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#Этап 1: Исследование данных (Самостоятельный EDA)
#Этап 2: Проектирование признаков (Feature Engineering)
#Этап 3: Построение пайплайна обучения (Modeling & Pipeline)
#Этап 4: Разработка веб-интерфейса (Streamlit)
df = pd.read_csv('movies_dataset.csv')
df = df[df['budget'] > 1000]
df['genres'] = df['genres'].fillna('None')
df['release_date'] = df['release_date'].fillna(df['release_date'].mode()[0])
df = df.drop('id', axis=1)
print(df.info())
plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x = 'budget',y = 'revenue') # влияние бюджета на доход небольшое
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x = 'popularity',y = 'revenue') #популярность не зависит от дохода
plt.show()
plt.figure(figsize=(10,6))
sns.scatterplot(data = df,x = 'vote_average',y = 'revenue') # средняя оценка не влияет на доход
plt.show()

# df['ROI'] = df['revenue'] / df['budget']
# print(df['ROI'].head())
# df['ROIDN'] = (df['ROI'] >= 2).astype(int) #окупился ли фильм
# print(df['ROIDN'].head())

x = df.drop('revenue', axis=1)
y = df['revenue']
numeric_features = ['budget','vote_average', 'vote_count','popularity'
]
categorical_features = ['release_date','original_language','genres']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

pipeline1 = Pipeline([
    ('preprocessor', preprocessor),
    ("model", RandomForestRegressor(random_state=42)),
])


pipeline2 = Pipeline([
    ('preprocessor', preprocessor),
    ("model", XGBRegressor(random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline1.fit(X_train, y_train)
pipeline2.fit(X_train, y_train)
ypred1 = pipeline1.predict(X_test)
ypred2 = pipeline2.predict(X_test)
print('RandomForestRegressor:')
print(ypred1)
print('mean_absolute_error:', mean_absolute_error(y_test, ypred1))
print('r2_score:', r2_score(y_test, ypred1))
print('XGBoost:')
print(ypred2)
print('mean_absolute_error:', mean_absolute_error(y_test, ypred2))
print('r2_score:', r2_score(y_test, ypred2))

param_grid1 = {
    "model__n_estimators": [5,25,100,200],
    "model__max_depth": [3,10,25,None]
}

param_grid2 = {
    "model__n_estimators": [10,25,50],
    "model__max_depth": [3,5,10],
    'model__learning_rate': [0.01,0.05,0.1]
}

gridsearch1 = GridSearchCV(pipeline1, param_grid1, cv=5, n_jobs=-1)
gridsearch2 = GridSearchCV(pipeline2, param_grid2, cv=5, n_jobs=-1)
gridsearch1.fit(X_train, y_train)
gridsearch2.fit(X_train, y_train)
print('Лучшие параметры для RandomForest:',gridsearch1.best_params_)
print('Лучшие параметры для XGBoost',gridsearch2.best_params_)
print("Качество кросс-валидации RandomForest",gridsearch1.best_score_)
print('Качество кросс-валидации XGBoost',gridsearch2.best_score_)
bestmodel1 = gridsearch1.best_estimator_
bestmodel2 = gridsearch2.best_estimator_
importances1 = bestmodel1.named_steps['model'].feature_importances_
importances2 = bestmodel2.named_steps['model'].feature_importances_
featurenames1 = bestmodel1.named_steps['preprocessor'].get_feature_names_out()
featurenames2 = bestmodel2.named_steps['preprocessor'].get_feature_names_out()
importancesdf1 = pd.DataFrame({'feature': featurenames1, 'importance': importances1})
importancesdf1 = importancesdf1.sort_values('importance', ascending=False)
importancesdf2 = pd.DataFrame({'feature': featurenames2, 'importance': importances2})
importancesdf2 = importancesdf2.sort_values('importance', ascending=False)
print(importancesdf1.head(5))
print(importancesdf2.head(5))

if gridsearch1.best_score_ > gridsearch2.best_score_:
    bestpipline = bestmodel1
    model_name = 'RandomForest'
else:
    bestpipline = bestmodel2
    model_name = 'XGBoost'
print(f'\n Сохраняем лучшую модель{model_name}')
joblib.dump(bestpipline, "best_movies_pipeline.joblib")
print('Модель успешно сохранена в файл best_movies_pipeline.joblib')
