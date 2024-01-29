# Week 5 Batch Processing

- Most of time, data ingestion job is a batch processing job (around 80%)
- PySpark is a good tool for batch processing for large amount of data and scalable

## Topic
1. First Look PySpark
2. Spark DataFrames
    - (2.5 Bash script downloading the data)
3. Spark SQL
4. Joins in Spark

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
    - Usually spark doesn't know the right schema like the pandas does, so utilzing pandas for referring the schema of the dataset is a good idea.
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
    
    # write output
    df\
        .repartition(4)\
        .write
        .parquet("output_path/") 

    # the output will be in form of "part-0001-...-.snappy.parquet"
    ```
## 3. Spark SQL (SQL with Spark)
