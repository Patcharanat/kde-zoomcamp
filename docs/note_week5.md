# Week 5 Batch Processing

- Most of time, data ingestion job is a batch processing job (around 80%)
- PySpark is a good tool for batch processing for large amount of data and scalable

## Topic
1. [First Look PySpark](#1-first-look-pyspark)
2. [Spark DataFrames](#2-spark-sql-and-dataframes)
    - 2.5 [Preparing data (Bash script downloading the data)](#25-preparing-data)
3. [Spark SQL](#3-spark-sql-sql-with-spark)
4. [Anatomy of a Spark Cluster](#4-anatomy-of-a-spark-cluster)
    - 4.1 [GroupBy in Spark](#41-groupby-in-spark)
    - 4.2 [Joins in Spark](#42-joins-in-spark)
5. [Resilient Distributed Datasets (RDD)](#5-resilient-distributed-datasets-rdd)
6. [Spark with Cloud](#6-spark-with-cloud)
    - 6.1 [Connecting to GCS](#61-connecting-to-gcs)
    - 6.2 [Creating a Local Spark Cluster](#62-creating-a-local-spark-cluster)
    - 6.3 [Setting up a Dataproc Cluster](#63-setting-up-a-dataproc-cluster)

## 1. First Look PySpark
- Repartition: to increase the number of output file partitions -> good for scaling processing node (parallelism)

## 2. Spark SQL and DataFrames
- PySpark introduce flexibility to custom logic to apply to the data more than just sql transformation.
- Actions vs. Transformations (lazy execution)
- Transformations (lazy)
    - Selecting columns
    - Filtering
    - Joins
    - Group by
- Actions (eager)
    - Show, take, head, count
    - write
- Pyspark basic syntax
    ```python
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    # initialize spark session
    spark = SparkSession.builder.appName("Some session name").getOrCreate()
    df = spark.read.options(header=True).csv("file_path.csv") # or parquet or json whatever

    # apply function to column
    new_df = df.withColumn("some_new_column", F.some_method(df.some_old_column))
    ```
- user defined function (udf)
    - When business requirement become more complex, and SQL is not enough. python is a better option, but more cumbersome also.
    - we write regular python defined function (def) to be able to unit test it, but when it come to be used in pyspark.
    ```python
    from pyspark.sql import functions as F
    from pyspark.sql.types import *
    
    # needed to be applied to a column
    def some_fn(argx):
        return argx[1:]//3 % 7 == True

    some_fn_udf = F.udf(some_fn, returnType=types.BooleanType())

    # to use in spark dataframe (df)
    output = df.withColumn("new_column_name", some_fn_udf(df.target_column))
    ```

### 2.5 Preparing data
- Example of Bash script downloading the data
    ```bash
    # download_data.sh

    # /yellow_tripdata_2021-01.csv
    TAXI_TYPE=$1 #"yellow"
    YEAR=$2 #2020

    URL_PREFIX="https://s3.amazonaws.com/nyc-tlc/trip+data"

    # for loop bash range
    for MONTH in {1..12}: do
        FMONTH=`printf "%02d" ${MONTH}`

        URL="${URL_PREFIX}/${TAXI_TYPE}_tripdata_${YEAR}-${FMONTH}.csv"

        LOCAL_PREFIX="data/raw/${TAXI_TYPE}/${YEAR}/${MONTH}"
        LOCAL_FILE="${TAXI_TYPE}_tripdata_${YEAR}_${FMONTH}.csv"
        LOCAL_PATH="${LOCAL_PREFIX}/${LOCAL_FILE}"

        mkdir -p ${LOCAL_PREFIX}

        # echo wget ${FMONTH} -O ${LOCAL_PATH}
        wget ${FMONTH} -O ${LOCAL_PATH}

        # compress file
        gzip ${LOCAL_PATH}
    done

    # to execute
    # ./download_data.sh green 2021
    ```
    - Some logging is nice to have (with echo)
- Tips to refer schema
    - Usually spark doesn't know the right schema like the pandas does, so utilzing pandas for referring the  (or generate schema) of the dataset is a good idea.
    ```python
    from pyspark.sql import SparkSession
    import pandas as pd
    
    # pandas can open compress file (.gz) without any extraction
    df_pd = pd.read_csv("/data/some_path.csv.gz", nrows=1_000)
    
    # get schema
    raw_schema = spark.createDataFrame(df_pd).schema
    print(raw_schema) # copy this and format the StructType (+add quote + add [] + import pyspark.sql.types + capital boolean)

    # from pyspark.sql.types import *
    # clean_schema = StructType([
    #     StructField("column1", IntegerType(), True),
    #     ...
    # ])

    # read data with schema
    df = spark.read\
            .option("header", "true")\
            .schema(clean_schema)\
            .csv("./data/")
            # we can use "path/*/*" to nested read the data in a folder
    
    # write output
    df\
        .repartition(4)\
        .write
        .parquet("output_path/") 

    # the output will be in form of "part-0001-...-.snappy.parquet"
    # in contrast, 
    df.coalesce(1).write.parquet("path/", mode='overwrite')
    # is used for reduce number of partitions
    ```
## 3. Spark SQL (SQL with Spark)
```python
# from pyspark.sql import SparkSession
# init session
# spark = SparkSession.builder.master(...).appName(...).getOrCreate()
# df = spark.read.option(...).parquet(...)

# This will raise an error of not knowing table name
spark.sql("""
SELECT * FROM df LIMIT 10;
""").show()

# you have to 'registerTempTable first'
df.registerTempTable("table_sql")
# and then it will work
spark.sql("""
SELECT * FROM df_sql LIMIT 10;
""").show()

# to save output from spark sql statement
df_result = spark.sql("""
SELECT * FROM df_sql WHERE ...;
""")

df_result.write.parquet("./data/sql_output/")
```
- There's some use-case, we should not have to use spark (especially, when sql is enough for data manipulating)
- But, sometimes we don't always have tools like hive, or presto, and if we already have spark cluster. So why don't we use it just executing sql query? it highly depends on case-by-case.

## 4. Anatomy of a Spark Cluster
- Spark Cluster, Driver, Master, Executors + cloud storage (Hadoop/HDFS)
    - Driver: submit spark job to a spark cluster
    - Cluster:
        - Master: distribute workload to executors
        - Executors: do the job

### 4.1 GroupBy in Spark
- Grouping by with spark has 2 stage in operation including:
    - Intermediate Result
        - (operate group by within partition)
        - think of we have multiple partitions of parquet file
    - Reshuffling (Implemented by algorithm: External merge sort)
        - (aggregated each groupby (with keys) into one partition for each key)
        - **The same key end up in the same partition**
        - one partition does have to contain only one key, but every single key must be contain only in its partition
        - This is expensive operation, using algorithm to sort data into partitions according to number of keys
        - Since, we don't need to produce that much cost if the file is not that large, **repartitioning** making it become more optimized in processing.
    
### 4.2 Joins in Spark
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
                    .master("local[*]") \
                    .appName("test") \
                    .getOrCreate()

df1 = spark.option().read.parquet("path1/")
df2 = spark.option().read.parquet("path2/")

# joining
df_join = df1.join(df2, on=["key1", "key2"], how="outer") # this line is lazy execution
```
- Joining in Spark also use reshuffling (***the same key end up in the same partition***) as same as grouping by in Spark
- **Broadcasting** in Joining happens when a large dataset joined with a small dataset making the small dataset broadcasted to every executors, resulting in one stage of joining operation with reshuffling removed. It gives a better performance (faster by one stage).
- we save some intermediate output to reduce processing costs when we need to replicate work, we called it **materialize**.

## 5. Resilient Distributed Datasets (RDDs)
- Spark is built on top of *RDD* which is involved to Spark low-level operation, but we use spark via API, so the abstraction of RDD is lesser important in usage but not useless to know.
- Dataframe has a schema but, RDD has collection of objects.
    - .rdd attribute in spark represents data in a list of rows
- Map and Reduce are work within RDD layer
    - map -> map the value (element) to the key
    - reduce -> reduce multiple elements in the same key to one element per key (e.g. groupby)
- mapPartition
    - partition (RDD) -> mapPartition -> partition (RDD)

## 6. Spark with Cloud

### 6.1 Connecting to GCS
- copying the local file to GCS with bash
    ```bash
    # to copy folder, use '-r' flag means recursive
    # to upload multiple file, use '-m' means multi-threaded (parallel) using all cpu core
    gsutil -m cp -r local_folder/ gs://bucket_name/path/target_folder
    ```
- connecting to GCS with Spark in **LOCAL**
    - using ".jar" file to specifically tell spark how to connect with GCP
    - gcs/gcs-connector/hadoop
    ```bash
    gsutil cp gs://hadoop-lib/gcs/gcs-connector-hadoop3-X.X.X.jar .
    ```
    - spark 
    ```python
    from pyspark.sql import SparkSession
    from pyspark.conf import SparkConf
    from pyspark.context import SparkContext

    credentials_location = "local/path/to/gcp_credentials.json"
    
    # setting config
    conf = SparkConf() \
        .setMaster('local[*]') \
        .setAppName('test') \
        .set("spark.jars", "./lib/gcs-connector-hadoop3-2.2.5.jar") \
        .set("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .set("spark.hadoop.google.cloud.auth.service.account.json.keyfile", credentials_location)

    sc = SparkContext(conf=conf)

    hadoop_conf = sc._jsc.hadoopConfiguration()

    hadoop_conf.set("fs.AbstractFileSystem.gs.impl",  "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS")
    hadoop_conf.set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    hadoop_conf.set("fs.gs.auth.service.account.json.keyfile", credentials_location)
    hadoop_conf.set("fs.gs.auth.service.account.enable", "true")

    # initiate session
    spark = SparkSession.builder \
            .config(conf=sc.getConf()) \
            .getOrCreate()

    df = spark.read.parquet('gs://bucket_name/pq/green/*/*')

    df.count()
    ```
- But, when we use google managed service for spark, we don't have manually config .jar file
 
### 6.2 Creating a Local Spark Cluster



### 6.3 Setting up a Dataproc Cluster