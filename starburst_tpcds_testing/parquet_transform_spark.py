import os
import json
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException
from termcolor import colored
import sys
from pyspark.sql.functions import monotonically_increasing_id, expr

# Load schema definitions
schema_path = os.path.join(os.path.dirname(__file__), "tpcds_schema.json")
with open(schema_path, 'r') as f:
    TABLE_SCHEMAS = json.load(f)

# Resolve paths relative to the script's location
script_dir = os.path.dirname(__file__)
input_dir = os.path.join(script_dir, "test_data/raw_files")
output_dir = os.path.join(script_dir, "test_data/parquet")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def handle_error(message):
    """Handle errors by printing in red and stopping the process."""
    print(colored(f"ERROR: {message}", "red"))
    print("Please use the cleanup tool before beginning again.")
    sys.exit(1)

def calculate_partitions(file_path, base_partition_size=128 * 1024 * 1024):
    """
    Calculate the number of partitions based on file size.
    :param file_path: Path to the file.
    :param base_partition_size: Target size of each partition in bytes (default: 128MB).
    :return: Number of partitions.
    """
    file_size = os.path.getsize(file_path)
    return max(1, file_size // base_partition_size)

def process_file(spark, file_path):
    """Process a single .dat file and convert it to Parquet."""
    try:
        # Read the .dat file; TPC-DS files are pipe-delimited with no header
        df = spark.read.csv(file_path, sep="|", header=False)

        # Get table name from file name (remove .dat extension)
        table_name = os.path.splitext(os.path.basename(file_path))[0]

        # Apply schema if available
        if table_name in TABLE_SCHEMAS:
            expected_columns = TABLE_SCHEMAS[table_name]["columns"]
            if len(df.columns) == len(expected_columns):
                df = df.toDF(*expected_columns)
                print(f"Applied schema for table {table_name}")
            else:
                print(f"Warning: Column count mismatch for {table_name}. Expected {len(expected_columns)}, got {len(df.columns)}")
        else:
            print(f"Warning: No schema found for table {table_name}")

        # Calculate the number of partitions based on file size
        num_partitions = calculate_partitions(file_path)

        # Add a partitioning column (e.g., partition_key) based on row index
        df = df.withColumn("partition_key", expr(f"monotonically_increasing_id() % {num_partitions}"))

        # repartition
        df = df.repartition(num_partitions)
        # also is recommended to increase the global JAVA memory heap allocation an system wide:
        # export _JAVA_OPTIONS="-Xmx500g"

        # Write DataFrame to Parquet with partitioning
        output_folder = os.path.join(output_dir, table_name)  # Remove .parquet from folder name
        df.write.mode("overwrite").partitionBy("partition_key").parquet(output_folder)

        # Remove the partitioning column after writing
        df = df.drop("partition_key")

        print(f"Successfully converted {file_path} to {output_folder} with {num_partitions} partitions")
    except AnalysisException as e:
        handle_error(f"Failed to process {file_path} due to schema mismatch: {e}")
    except Exception as e:
        handle_error(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parquet transform")
    parser.add_argument("--custom_dir", required=False, type=str, help="Directory for temporary files location, instead of default /tmp directory")
    args = parser.parse_args()

    # Initialize Spark session
    spark = SparkSession.builder \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.executor.memoryOverhead", "2g") \
        .appName("TPC-DS Parquet Transformer") \
        .master("local[*]")
    
    if args.custom_dir != "":
        spark = spark.config("spark.local.dir", args.custom_dir)
    
    spark = spark.getOrCreate()

    # Get list of all .dat files in the input directory
    dat_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".dat")]

    if not dat_files:
        print("No .dat files found in the input directory.")
    else:
        # Process files sequentially on the driver node
        for file_path in dat_files:
            process_file(spark, file_path)

    # Stop the Spark session
    spark.stop()