# Week 2 - Document

## Data lake Introduction

- Data lake vs Data warehouse
    - more unorganized (no need to define relationships or write the schema)
    - faster, and cheaper
    - Data swamp → No useful data for further usage **such as** can’t join, or different file type
- ETL vs ELT
    - ETL - Small data → ready for further use (organized)
    - ELT - Large data → need further processes before usage
- Cloud provider
    - GCP - Google Cloud - Cloud Storage
    - AWS - Amazon Web Service - S3
    - Azure - Microsoft Azure - Azure Blob

## Introduction to workflow orchestration (Prefect)

- orchestration: governing data flow respecting orchestration rules and business logic
- data flow: binding disparate sets of applications together, so they can run schedule
- Core of orchestration
    - Remote execution
    - scheduling
    - retries
    - caching
    - integrated with external system (APIs, databases)
    - Ad-hoc runs
    - Parameterization
    - Alerting when something fail

*all about Prefect deployment is skipped*

## DE Zoomcamp 2022 Using Airflow

- Popular orchestration tools: Airflow, Prefect
- Airflow consist of 3 main component
    - Webserver: UI
    - Scheduler (Executor)
    - Metadata Database (backend airflow environment)
- Setting up Airflow
    1. create sub-directory `airflow` at the current project dir
    2. set Airflow user: do in GitBash in airflow directory
        
        ```bash
        mkdir -p ./dags ./logs ./plugins
        echo -e "AIRFLOW_UID=$(id -u)" > .env
        ```
        
        or create .env file and then type in “AIRFLOW_UID=50000”
        
    3. import the latest official setup template `docker-compose.yml` file
        
        ```bash
        curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
        ```
        
        which contain a lot of services defined. check cleaned version of docker-compose.yml
        
        [docker-compose_cleaned.yml](https://www.notion.so/docker-compose_cleaned-yml-3f5f3e39879a4e44bfa6d6b12bb648fa?pvs=21)
        
- Ingest Data into postgres database
    - Writing DAG
    - make it scheduling and parameterizing (accept different url or save different file name)
    - connect with Postgres database ( create_engine → connect() → load by chuck )
    - if run the docker-compose file separately, we need to use the network to make containers communicable

## Transfer Service

- Data Transfer in gcp
    - it can transfer data from S3 (aws) or azure blob (microsoft) to gcs
        - Note: to transfer s3 to gcs, we need access key → get access key from aws site
    - we are able to set to make transferring scheduling (not recommend due to cost)
    - config like creating a new bucket
- done by
    - GCP UI → `Data Transfer`
    - Terraform → `Terraform google storage transfer job`