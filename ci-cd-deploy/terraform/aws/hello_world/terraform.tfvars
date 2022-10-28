hello_minio_aws_region  = "us-east-1"

hello_minio_aws_vpc_cidr_block = "10.0.0.0/16"
hello_minio_aws_vpc_cidr_newbits = 4

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

hello_minio_aws_eks_cluster_name = "hello_minio_aws_eks_cluster"
hello_minio_aws_eks_cluster_endpoint_private_access = true
hello_minio_aws_eks_cluster_endpoint_public_access = true
hello_minio_aws_eks_cluster_public_access_cidrs = ["0.0.0.0/0"]
hello_minio_aws_eks_node_group_name = "hello_minio_aws_eks_node_group"
hello_minio_aws_eks_node_group_instance_types = ["t3.large"]
hello_minio_aws_eks_node_group_desired_size = 3
hello_minio_aws_eks_node_group_max_size = 5
hello_minio_aws_eks_node_group_min_size = 1