import tiledb
import numpy as np
import os, shutil

# Local path
array_local = os.path.expanduser("./tiledb_demo")

# Create a simple 1D array
tiledb.from_numpy(array_local, np.array([1.0, 2.0, 3.0]))

# Read the array
with tiledb.open(array_local) as A:
	print(A[:])