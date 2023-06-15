from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


def _extract_data():
    return


with DAG(
    dag_id="Ingest-nyc-taxi",
    start_date=datetime(2023, 6, 14),
    schedule=None
) as dag:
    
    wget_test = BashOperator(
        task_id='echo',
        bash_command='echo Initiating stage'
    )

    extract_data = PythonOperator(
        task_id='extract_data',
        python_callable=_extract_data
    )

    wget_test >> extract_data