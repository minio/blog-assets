variable "hello_minio_aws_region" {
  description = "Hello MinIO AWS region"
  type        = string
}

/* */

variable "hello_minio_aws_vpc_cidr_block" {
  description = "Hello MinIO AWS VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "hello_minio_aws_vpc_cidr_newbits" {
  description = "Hello MinIO AWS VPC CIDR new bits"
  type        = number
  default     = 4
}

/* */

variable "hello_minio_public_igw_cidr_blocks" {
  type = map(number)
  description = "Hello MinIO Availability Zone CIDR Mapping for Public IGW subnets"
 
  default = {
    "us-east-1b" = 1
    "us-east-1d" = 2
    "us-east-1f" = 3
  }
}

/* */

variable "hello_minio_private_ngw_cidr_blocks" {
  type = map(number)
  description = "Hello MinIO Availability Zone CIDR Mapping for Private NGW subnets"
 
  default = {
    "us-east-1b" = 4
    "us-east-1d" = 5
    "us-east-1f" = 6
  }
}

/* */

variable "hello_minio_private_isolated_cidr_blocks" {
  type = map(number)
  description = "Hello MinIO Availability Zone CIDR Mapping for Private isolated subnets"
 
  default = {
    "us-east-1b" = 7
    "us-east-1d" = 8
    "us-east-1f" = 9
  }
}

/* */

variable "hello_minio_aws_eks_cluster_name" {
  description = "AWS EKS Cluster name"
  type        = string
  default     = "hello_minio_aws_eks_cluster"
}

variable "hello_minio_aws_eks_cluster_endpoint_private_access" {
  description = "AWS EKS Cluster endpoint private access"
  type        = bool
  default     = true
}

variable "hello_minio_aws_eks_cluster_endpoint_public_access" {
  description = "AWS EKS Cluster endpoint public access"
  type        = bool
  default     = true
}

variable "hello_minio_aws_eks_cluster_public_access_cidrs" {
  description = "AWS EKS Cluster public access cidrs"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "hello_minio_aws_eks_node_group_name" {
  description = "AWS EKS Node group name"
  type        = string
  default     = "hello_minio_aws_eks_node_group"
}

variable "hello_minio_aws_eks_node_group_instance_types" {
  description = "AWS EKS Node group instance types"
  type        = list(string)
  default     = ["t3.large"]
}

variable "hello_minio_aws_eks_node_group_desired_size" {
  description = "AWS EKS Node group desired size"
  type        = number
  default     = 3
}

variable "hello_minio_aws_eks_node_group_max_size" {
  description = "AWS EKS Node group max size"
  type        = number
  default     = 5
}

variable "hello_minio_aws_eks_node_group_min_size" {
  description = "AWS EKS Node group min size"
  type        = number
  default     = 1
}