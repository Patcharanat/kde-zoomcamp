#!/usr/bin/env python
# coding: utf-8

import pyarrow.parquet as pq
from sqlalchemy import create_engine
import pandas as pd

if __name__=='__main__':
    trips = pq.read_table("./data/yellow_tripdata_2023-01.parquet")
    trip = trips.to_pandas()

    sample = trip.head(100)
    sample.to_csv('./data/sample_yellow_taxi_2023.csv', index=False)

    engine = create_engine('postgresql://postgres:postgres@localhost:5432/ny_taxi')
    engine.connect()

    print(pd.io.sql.get_schema(sample, name='yellow_taxi_data'))

    df_iter = pd.read_csv('./data/sample_yellow_taxi_2023.csv')

    df = next(df_iter)

    print(len(df))

    df.to_sql(name='yellow_taxi_data', con=engine, if_exists='append')