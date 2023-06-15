# Week 4 - Document

## Analytics Engineering

- Role: DE + DA
- Tooling: Data Loading → Data Storing → Data modeling → Data presentation
- ETL vs ELT
- Dimensional Modeling (Star Schema)
    - Architecture of Dimensional Modeling
        - Stage Area
        - Processing Area
        - Presentation Area

## dbt (data build tools)

- SQL for analytics
- dbt Core & Cloud
    - local open source & SaaS app
    - local postgres database can’t be connected with dbt cloud
- dbt from scratch
    - dbt provide `starter project` with all basic folders and files
        - `dbt_project.yml` to config dbt project
        - dbt local
            - working at needed directory then:
                ```bash 
                dbt init
                ```