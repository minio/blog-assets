hello_minio_aws_region  = "us-east-2"

hello_minio_aws_vpc_cidr_block = "10.2.0.0/16"
hello_minio_aws_vpc_cidr_newbits = 4

hello_minio_public_igw_cidr_blocks = {
  "us-east-2a" = 1
  "us-east-2b" = 2
  "us-east-2c" = 3
}

hello_minio_private_ngw_cidr_blocks = {
  "us-east-2a" = 4
  "us-east-2b" = 5
  "us-east-2c" = 6
}

hello_minio_private_isolated_cidr_blocks = {
  "us-east-2a" = 7
  "us-east-2b" = 8
  "us-east-2c" = 9
}

hello_minio_aws_instance_key_name="aj_terraform"
