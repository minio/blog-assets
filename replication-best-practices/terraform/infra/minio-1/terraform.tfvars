hello_minio_region  = "us-east-1"

hello_minio_cidr_block = "10.0.0.0/16"
hello_minio_cidr_newbits = 4

hello_minio_public_igw_cidr_blocks = {
  "us-east-1b" = 1
  "us-east-1d" = 2
  "us-east-1f" = 3
}

hello_minio_private_ngw_cidr_blocks = {
  "us-east-1b" = 4
  "us-east-1d" = 5
  "us-east-1f" = 6
}

hello_minio_private_isolated_cidr_blocks = {
  "us-east-1b" = 7
  "us-east-1d" = 8
  "us-east-1f" = 9
}

hello_minio_instance_key_name="aj_terraform"
