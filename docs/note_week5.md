# Week 5 Batch Processing

- Most of time, data ingestion job is a batch processing job (around 80%)
- PySpark is a good tool for batch processing for large amount of data and scalable

## 1. First Look PySpark
- Repartition: to increase the number of output file partitions -> good for scaling processing node (parallelism)

## Spark DataFrames
- Actions vs. Transformations (lazy execution)
- Transformations (lazy)
    - Selecting columns
    - Filtering
    - Joins
    - Group by
- Actions (eager)
    - Show, take, head, count
    - write