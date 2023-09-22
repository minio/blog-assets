import duckdb

def query_minio_duckdb(bucket_name, object_name):
    try:
        # Connect to DuckDB and read Parquet files
        conn = duckdb.connect()
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
        conn.execute("SET s3_region='us-east-1'")
        conn.execute("SET s3_url_style='path'")
        conn.execute("SET s3_endpoint='minio:9000'")
        conn.execute("SET s3_access_key_id='***'")
        conn.execute("SET s3_secret_access_key='***'")
        conn.execute(f"CREATE TABLE data AS SELECT * FROM read_parquet('s3://{bucket_name}/{object_name}')")

        # Number of Rows in data
        total_rows_result = conn.execute("SELECT COUNT(*) AS TotalRows FROM data")
        total_rows = total_rows_result.fetchone()[0]
        print(f"Number of rows in data: {total_rows}")

        # Make a list of columns
        columns_result = conn.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'data'")
        columns = columns_result.fetchall()

        # Number of nulls in each column
        for column in columns:
            column_name = column[0]
            result = conn.execute(f"SELECT COUNT(*) FROM data WHERE {column_name} IS NULL")
            null_count = result.fetchone()[0]
            print(f"Number of nulls in '{column_name}': {null_count}")

        # Summary statistics for numeric columns
        numeric_columns_result = conn.execute("SELECT column_name FROM information_schema.columns"
                                              " WHERE table_name = 'data' "
                                              "AND data_type IN ('BIGINT','INTEGER','DOUBLE')")
        numeric_columns = numeric_columns_result.fetchall()
        for column in numeric_columns:
            column_name = column[0]
            result = conn.execute(
                f"SELECT MIN({column_name}), MAX({column_name}), AVG({column_name}), STDDEV({column_name}) FROM data")
            min_val, max_val, avg_val, stddev_val = result.fetchone()
            print(f"Summary statistics for '{column_name}':")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
            print(f"  Average: {avg_val}")
            print(f"  Standard Deviation: {stddev_val}")

        # Clean Up
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


# Replace 'ducknest' and 'insects.parquet' with your bucket and object names
query_minio_duckdb('ducknest', 'insects.parquet')
