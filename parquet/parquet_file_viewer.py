import pandas as pd
import pyarrow.parquet as pq

def print_parquet(file_path, num_rows=5):
    # Read Parquet file into a pyarrow Table
    table = pq.read_table(file_path)

    # Extract a Pandas DataFrame from the Table
    df = table.to_pandas()

    # Set display options to show all columns without truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Print the first few rows of the DataFrame
    print(df.head(num_rows))

# Specify the path to your Parquet file
parquet_file_path = "path/to/your/file.parquet"

# Print the first 5 rows of the Parquet file with tidy column display
print_parquet(parquet_file_path)