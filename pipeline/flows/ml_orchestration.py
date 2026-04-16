#!/usr/bin/env python
# coding: utf-8

import mlflow
mlflow.set_experiment("mlops-learn1")
mlflow.autolog()
if mlflow.active_run() is not None:
    mlflow.end_run()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from prefect import flow, task

# Load and clean data
@task
def load_data():
    playerStats = pd.read_csv("/home/nas/portfolio/portfolio-ml/data/playerStats.csv")
    playerStats.drop(columns=["Player Name", "Team"],inplace=True)
    X=playerStats.drop(columns=["Position"])
    y=playerStats["Position"]
    X = X[["Age","Sets Per Match", "Receives Per Match", "Serves Per Match", "Blocks Per Match", "Digs Per Match", "Attacks Per Match"]]
    X = X.drop(columns=["Age", "Serves Per Match"])
    return X, y
    
# Encode and scale data
@task
def preprocess_data(X_train, X_test, y_train, y_test):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_encoded = scaler.transform(X_test)
    return X_train_scaled, y_train_encoded, X_test_encoded, y_test_encoded

# Split for training & testing
@task
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# @task
# def pipeline():


@task
def train_random_forest(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="RandomForest_best_model"):
        best_rf = RandomForestClassifier(
                bootstrap=True,
                ccp_alpha=0.0,
                class_weight=None,
                criterion="gini",
                max_depth=30,
                max_features="log2",
                max_leaf_nodes=None,
                max_samples=None,
                min_impurity_decrease=0.0,
                min_samples_leaf=4,
                min_samples_split=5,
                min_weight_fraction_leaf=0.0,
                monotonic_cst=None,
                n_estimators=443,
                n_jobs=None,
                oob_score=False,
                random_state=42,
                verbose=0,
                warm_start=False,
            )
        best_rf.fit(X_train, y_train)
        preds = best_rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(best_rf, "random_forest_model", registered_model_name="random_forest_model_dev", input_example=X_train)
        return best_rf
@flow
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, y_train_encoded, X_test_encoded, y_test_encoded = preprocess_data(X_train, X_test, y_train, y_test)
    best_rf = train_random_forest(X_train_scaled, y_train_encoded, X_test_encoded, y_test_encoded)
    preds = best_rf.predict(X_test_encoded)
    acc = accuracy_score(y_test_encoded, preds)
    f1 = f1_score(y_test_encoded, preds, average="weighted")
    print(f"Accuracy: {acc}")  
    print(f"F1 Score: {f1}")
    
if __name__ == "__main__":
    main()