import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

def load_data(path):
    return pd.read_csv(path)

def handle_missing_values(data, columns):
    imputer = SimpleImputer(strategy='most_frequent')
    for column in columns:
        data[column] = imputer.fit_transform(data[[column]])
    return data

def encode_categorical_variables(data, categorical_columns):
    label_encoder = LabelEncoder()
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    return data

def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with Grid SearchCV
def train_xgboost_classifier(X_train, y_train):
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.3, 0.7]
    }
    model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False)
    grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=3, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    return grid_search.best_estimator_

# Naive Bayes
def train_naive_bayes(X_train, y_train):
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    return nb_clf

# Decision Tree
def train_decision_tree(X_train, y_train):
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    return dt_clf

def cross_validate_and_select_model(model, X_train, y_train, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    return roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

def prepare_test_data(test_data, X_train_columns):
    test_data = test_data.reindex(columns=X_train_columns, fill_value=0)
    return test_data

def generate_submission(model, test_data, test_ids, filename):
    test_predictions = model.predict_proba(test_data)[:, 1]
    submission = pd.DataFrame({'ID': test_ids, 'Prediction': test_predictions})
    submission.to_csv(filename, index=False)

path_train = 'train_final.csv/train_final.csv'
path_test = 'test_final.csv/test_final.csv'

data = load_data(path_train)
test_data = load_data(path_test)

# Preprocess data
columns_to_impute = ['workclass', 'occupation', 'native.country']
data = handle_missing_values(data, columns_to_impute)
test_data = handle_missing_values(test_data, columns_to_impute)

categorical_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
data = encode_categorical_variables(data, categorical_columns)
test_data = encode_categorical_variables(test_data, categorical_columns)

X_train, X_test, y_train, y_test = split_data(data, 'income>50K')

# Train models and evaluate
xgb_clf = train_xgboost_classifier(X_train, y_train)
nb_clf = train_naive_bayes(X_train, y_train)
dt_clf = train_decision_tree(X_train, y_train)

# Decision Tree with Grid Search
param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
best_dt_clf = cross_validate_and_select_model(dt_clf, X_train, y_train, param_grid)

print("XGB Classifier AUC:", evaluate_model(xgb_clf, X_test, y_test))
print("Naive Bayes AUC:", evaluate_model(nb_clf, X_test, y_test))
print("Decision Tree AUC:", evaluate_model(best_dt_clf, X_test, y_test))

# Preprocess and predict on test data
test_ids = test_data['ID']
test_data = prepare_test_data(test_data, X_train.columns)
generate_submission(best_dt_clf, test_data, test_ids, 'submissionDT.csv')
generate_submission(xgb_clf, test_data, test_ids, 'submissionXGB.csv')
generate_submission(nb_clf, test_data, test_ids, 'submissionNB.csv')