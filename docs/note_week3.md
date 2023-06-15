# Week 3 - Document

- OLAP vs OLTP
    - Online Analytical Process / Online Transaction Processing
    - OLTP → backend database, more normalized data (RDBM)
    - OLAP → data warehouse concept, more denormalized data, useful for DA/DS/BI
- What is data warehouse?
    - OLAP type
    - data mart: data warehouse which serve specific team unit
- Bigquery
    - Cost
        - the more data it process → the more it cost
        - On demand pricing & Flat Rate pricing
    - Partitioning and Clustering
        - the partitioned and clustered table make much lesser and lesser cost respectively compared with a normal table (Internal BQ management)
            
            ```sql
            -- In Bigquery
            CREATE OR REPLACE TABLE db.table_name_partitioned_clustered
            PARTITION BY -- e.g. DATE(some_column)
            CLUSTER BY -- e.g. some_another_column
            AS
            SELECT * FROM db.main_table;
            ```
            
            - Partitioning: WHERE DATE… is the same
            - Clustering: a same value in the same partition in a column stays adjacent to each other in the table
            - Bigquery Partition concept
                - Time-unit column
                - Ingestion time (_PARTITIONTIME)
                - Integer range partitioning
                - when using Time-unit or Ingestion time
                    - Daily (Default)
                    - Hourly
                    - Monthly, yearly
                - Number of partitions limit is 4000
            - Bigquery Clustering concept
                - columns we specify are used to colocate related data
                - The order of columns is important
                    - affect sorting order (sorting A → B → C respectively)
                - Clustering Improves
                    - Filter queries
                    - Aggregate queries
                - we can specify up to 4 clustering columns
                - clustering data types
                    - Date
                    - Bool
                    - Geography
                    - Int64
                    - Numeric
                    - Bignumeric
                    - string
                    - timestamp
                    - datetime
            - Table with data size < 1 GB, don’t show significant improvement with partitioning and clustering
            - To choose Clustering over Partitioning
                - when data < 1 GB or columns have a high amount of granularity
                - when number of partitions > 4000
    - Best practices
        - Avoid SELECT *
        - Price your queries before running them
        - Use clustered or partitioned tables
        - Use streaming inserts with caution
        - Materialize query results in stages
    - Query performance
        - Filter on partitioned columns (or clustered)
        - denormalizing data
        - use nested or repeated column
        - use external data source appropriately e.g. using data from GCS consume more cost
        - Reduce data before using a JOIN
        - Do not treat WITH clause as prepared statements
        - Avoid oversharing tables
    - Internals
        - skipped
    - ML in BQ
        - skipped