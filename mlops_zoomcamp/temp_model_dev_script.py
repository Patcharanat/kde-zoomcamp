import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb

from sklearn.metrics import classification_report

import pickle

df = pd.read_csv("./data/credit_risk_dataset.csv")

# EDA
df.info()
df.head()
# df["person_home_ownership"].value_counts()
# df["loan_intent"].value_counts()
# df["loan_grade"].value_counts()
# df["cb_person_default_on_file"].value_counts()

df = df.rename(columns={
    "person_age": "age",
    "person_income": "income",
    "person_home_ownership": "home_ownership",
    "person_emp_length": "employed_length_year",
    "loan_intent": "loan_purpose",
    # "load_grade": "load_grade"
    "loan_amnt": "loan_amount",
    "loan_int_rate": "interest_rate",
    "loan_status": "is_default",
    # "loan_percent_income": "loan_percent_income"
    "cb_person_default_on_file": "is_historical_default",
    "cb_person_cred_hist_length": "credit_history_length"
})

# sum(df["employed_length_year"].fillna(0).apply(lambda x: int(x)) != df["employed_length_year"].fillna(0))

# drop null
# filter out null 'employed_length_year'
df = df[~df["employed_length_year"].isnull()]
df = df[~df["interest_rate"].isnull()]
df = df.reset_index(drop=True)

# cast data type
df["employed_length_year"] = df["employed_length_year"].astype(int)

df.info()

# outlier removal
df.describe().boxplot()
plt.xticks(rotation=75)
plt.show()

# IQR
# define the upper and lower bound
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

# Create arrays of Boolean values indicating the outlier rows
upper_array = np.where(df['income'] >= upper)[0]
lower_array = np.where(df['income'] <= lower)[0]

# Removing the outliers
df = df.drop(index=upper_array)
df = df.drop(index=lower_array)

# Z-score
# assume data = DataFrame
z = np.abs(
    stats.zscore(
        df[["age", "income", "employed_length_year", "interest_rate", "loan_percent_income", "credit_history_length"]]
    )
)
threshold = 3
outliers = np.where(z > threshold)

# DataFrame with no oulier
# df = df[(z < threshold).all(axis=1)]
df = df[(z < threshold).all(axis=1)]

df = df.reset_index(drop=True)

df.describe().boxplot()
plt.xticks(rotation=75)
plt.show()

df.head()

# one-hot-encode
df["is_historical_default"] = df["is_historical_default"].map({
    "Y": 1,
    "N": 0
})

df["loan_grade"] = df["loan_grade"].map({
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7
})

df = pd.get_dummies(df, columns=["home_ownership", "loan_purpose"])
df.head()

# label encode
df.insert(0, "id", df.index)
df.head()

input_col = list(df.columns)
input_col.remove("is_default")
input = df[input_col]
output = df["is_default"]

X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=10)

# train model and validation
lr = ("LogisticRegression", LogisticRegression())
tree = ("DecisionTreeClassifier", DecisionTreeClassifier())
lgb_model = ("LGBMClassifier", lgb.LGBMClassifier(verbose=0))

trained_model = []
for model_name, model in [lr, tree, lgb_model]:
    print(f"Model: {model_name}")
    model.fit(X_train, y_train)
    trained_model.append(model)
    prediction = model.predict(X_test)
    print(classification_report(y_true=y_test, y_pred=prediction))
    print("------------------------------------------------------------------------------")


# mlflow
import mlflow
from mlflow.data.pandas_dataset import PandasDataset
import lightgbm as lgb

from sklearn.metrics import accuracy_score, f1_score

# set experiment tracking with mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit-risk-prediction")
# Create an instance of a PandasDataset
dataset = mlflow.data.from_pandas(
    df, 
    source="https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data",
    name="credit_risk_dataset", 
    targets="is_default"
)

with mlflow.start_run():
    mlflow.set_tag("developer", "patcha-ranat")   
    mlflow.log_input(dataset, context="training")
    # mlflow.log_params("val-data-path", "./data/credit_risk_dataset.csv")

    # model params
    # num_iterations = 100
    # mlflow.log_params("num_iterations", num_iterations)
    params = {
        "num_iterations": 100,
        "max_depth": 4,
        "num_leaves": 15
    }
    mlflow.log_params(params)
    
    # train
    lgb_model = lgb.LGBMClassifier(**params)
    lgb_model.fit(X_train, y_train)

    # predict
    y_pred = lgb_model.predict(X_test)

    # eval
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_macro_avg", f1)
    mlflow.lightgbm(lgb_model=lgb_model, artifact_path="model")

print(trained_model)

# save model
with open("light_gbm_v1.bin", "wb") as f_out:
    pickle.dump(trained_model[2], f_out)
f_out.close()

# save output
trained_model[2].predict(X_test)