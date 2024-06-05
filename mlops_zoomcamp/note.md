# MLOps Zoomcamp Note

*MLOps = a set of best practice to putting Machine Learning to production*

General 3 stages of Machine Learning Project
1. Design: consider if machine learning is suitable to solve the problem
2. Train: Experiment Models
3. Operate: Deployment, Monitor model's performance

## Table of Contents
- [1.2 Environment Preparation](#12-environment-preparation)
- [1.3 (Optional) Training a ride duration prediction model](#13-optional-training-a-ride-duration-prediction-model)
- [1.4 Course Overview](#14-course-overview)
- [1.5 MLOps Maturity Model](#15-mlops-maturity-model)
- [2.1 Experiment Tracking Intro](#21-experiment-tracking-intro-mlflow)
- [2.2 Getting started with MLflow](#22-getting-started-with-mlflow)
- [2.3 Experiment tracking with MLflow](#23-experiment-tracking-with-mlflow)

## 1.2 Environment Preparation

In the video, Alexey recommended to use linux operating system (EC2 in demonstration)
- Manually creating an instance on AWS cloud
- Creating "Key Pair", download it and place in ~/.ssh path
- specify instance's spec & launch
- Copying Public IP address then go to local terminal
    ```bash
    # username: ubuntu in demonstration
    ssh -i ~/.ssh/key_pair.extension <username>@<copied IP address>

    # connected to the remote instance
    ```
    - but we don't have to execute this bash command every time to connect to the remote instance, do this in local machine
    ```bash
    nano .ssh/config/
    
    # specify config
    Host <host-name>
        HostName <Public IP Address>
        User <Username>
        IdentityFile <Full Path to key-pair file location> 
        StrictHostKeyChecking no

    # then save and exit
    ssh <host-name>
    ```
- Installing py env (preparing python environment)
    ```bash
    wget <link to anaconda linux installer (.sh)>
    bash <downloaded .sh file>
    ```
- To open some service liek jupyter notebook, we might need to access via port such as "htpp://localhost:8888". We need to forward port before, which is very easy to do via vscode `PORT`.

## 1.3 (Optional) Training a ride duration prediction model
- Saving ML model
    ```python
    import pickle

    # from ... import model
    # model.fit(x_train, y_train)
    # dv = DictVectorizer().fit(...)

    with open("models/lin_reg.bin", "wb") as f_out:
        pickle.dump((dv, model), f_out)
    ```
- There're always DS' notebook where the model is developed is typically not well-arranged and not applicable in production.

## 1.4 Course overview
- Experiment Tracking and Model Management --> MLflow
- ML Pipeline --> Prefect & Kubeflow
    - Parameterized
- Model Deployment & Serving (used as a service)
- Monitoring and automatically Re-train model

## 1.5 MLOps Maturity Model
- Level-0: No MLOps 
    - No automation, All code in Jupyter Notebook (PoC)
- Level-1: DevOps, but No MLOps
    - What are included
        - Releases are automated
        - Unit & Integration Tests
        - CI/CD
        - OPs Metrics (Request workload & Network aspect)
    - What are not included
        - No Experiment Tracking
        - No Reproducibility
        - Data Scientists are separated from Engineers
- Level-2: Automated Training
    - Infrastructure
        - Trainning Pipeline
        - Experiment Tracking
        - Model Registry
    - Low Friction Deployment (maybe just bump version on platform)
    - DS works with Engineer Team
- Level-3: Automated Deployment
    - Easy to Deploy model
    - Full ML Pipeline: Data prep. -> Train Model -> Deploy model
    - A/b Testing
    - Model Monitoring
- Level-4: Full MLOps Automation
    - Automatic Train / Re-train / Deployment

## 2.1 Experiment Tracking Intro (MLflow)
- Install and how to use MLflow
- Important concept:
    - ML experiment: the process of building an ML model
    - Experiment run: each trial in an ML experiment
    - Run artifact: any file that is associated with an ML run
    - Experiment meta data
- Experiment tracking
    - Source code
    - Environment
    - Data
    - Model
    - Hyperparameters
    - Metrics
    - ...
    - For:
        - Reproducibility
        - Organization: finding information of this and that that we have developed before.
        - Optimization: tune to be a better ML model
- MLflow
    - open source platform for the machine learning lifecycle
    - A simple python package: `pip install mlflow`
    - components:
        - Tracking
        - Models
        - Model Registry
        - Projects
    - MLflow module allows you to organize your experiments into runs and keep track of
        - Parameters
        - Metrics
        - Metadata
        - Artifacts
        - Models
    
## 2.2 Getting started with MLflow
- To install dependencies that require for the project, you have to install it with `requirements.txt`, but the best practice is to use separated python environment such as conda or venv (virual env) to not mess up with packages installed in the local configuration.

```bash
conda create -n <environment-name> python=3.9

conda activate <environment-name>

# you will see something like (<environment-tracking-env> -> ...) in terminal
pip install -r requirements.txt

pip list

mlflow --version

# mlflow ui

# mlflow ui with backend database
mlflow ui --backend-store-uri sqlite:///mlflow.db

# then copy a given local ip address (and port) and open in the browser
```

*Note: **venv** vs **Conda***
| venv | conda |
|:-:|:-:|
| light weight | data-sci lib included as default|
| Pure Python Project | Accept Multi-language Project |

- To set mlflow tracker; go to notebook developing ML model
    ```python
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db") # it will try to save artifact to this db
    mlflow.set_experiment("custom-experiment-name")
    
    # develop your model in the notebook with mlflow lib
    with mlflow.start_run():
        
        mlflow.set_tag("developer", "dev_name") # this may be useful for a large team
        mlflow.log_params("train-data-path", "./path/to/data.parquet")
        mlflow.log_params("val-data-path", "./path/to/data.parquet")

        alpha = 0.1
        mlflow.log_params("alpha", alpha)
        
        lr = model(alpha)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
    ```
    - then go check experiment in mlflow UI 


## 2.3 Experiment tracking with MLflow
