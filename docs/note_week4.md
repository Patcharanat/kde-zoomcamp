# Week 4 - Document

## Analytics Engineering

- Role: DE + DA
- Tooling: Data Loading → Data Storing → Data modeling → Data presentation
- ETL vs ELT
    - ETL:
        - More stable for data analysis (More cleaned)
        - Higher storage & computational cost
    - ELT:
        - Faster, and more flexible for data analysis
        - lower cost, lower maintenance
- Facts vs Dimensions
- Dimensional Modeling (Star Schema)
    - Architecture of Dimensional Modeling
        - Stage Area
        - Processing Area
        - Presentation Area

## dbt (data build tools)

- Made the software engineering practice become available to SQL code by adopting versioning, unit testing, CI/CD, modularity, portability, etc. (+ doc generation, data lineage visualization)
- dbt comply against data warehouse, turning raw table from data warehouse into user's specified data model, transforming, and persist into data warehouse to finish the process.
- dbt Core & Cloud
    - local open source & SaaS app
    - local postgres database can’t be connected with dbt cloud
- dbt from scratch
    - dbt provide `starter project` with all basic folders and files
        - [`dbt_project.yml`](../dbt/kde_dbt/dbt_project.yml) to config dbt project
        - dbt local
            - working at needed directory then:
                ```bash 
                # clone the starter project
                dbt init
                ```
        - dbt cloud
            - cloud.getdbt -> put in sub-directory to generate working space
    - If you're using dbt Core, you'll need a [profiles.yml](../dbt/profiles.yml) file that contains the connection details for your data platform. When you run dbt Core from the command line, it reads your dbt_project.yml file to find the profile name, and then looks for a profile with the same name in your profiles.yml file. This profile contains all the information dbt needs to connect to your data platform.
    - ***profiles.yml*** should be located in "$HOME/.dbt/profiles.yml" defining the dbt connection

## Build the First dbt Models
- Modular data modeling
- We write dbt, we write it in SELECT statement (no DDL or DML)
    - when dbt comply the code, it will transform our SQL code to DDL statement following these **materialization** strategies
        - Table: physical database
        - View: virtual database (output can be changed depending on the source at a time)
        - Incremental: physical storage
            - create the table with SELECT statement
            - insert only new data into the table
        - Ephemeral (very much like CTE in SQL)

    