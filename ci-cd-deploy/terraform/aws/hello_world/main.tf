terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.31.0"
    }
  }

  backend "s3" {
    bucket = "aj-terraform-bucket"
    key    = "tf/aj/mo"
    region = "us-east-1"
  }

}

provider "aws" {
  region  = var.hello_minio_aws_region
}

module "hello_minio_aws_vpc" {
  source = "../modules/vpc"

  minio_aws_vpc_cidr_block   = var.hello_minio_aws_vpc_cidr_block
  minio_aws_vpc_cidr_newbits = var.hello_minio_aws_vpc_cidr_newbits

  minio_public_igw_cidr_blocks       = var.hello_minio_public_igw_cidr_blocks
  minio_private_ngw_cidr_blocks      = var.hello_minio_private_ngw_cidr_blocks
  minio_private_isolated_cidr_blocks = var.hello_minio_private_isolated_cidr_blocks

}

/* EKS Example */

module "hello_minio_aws_eks_cluster" {
  source = "../modules/eks"

  minio_aws_eks_cluster_name                    = var.hello_minio_aws_eks_cluster_name
  minio_aws_eks_cluster_endpoint_private_access = var.hello_minio_aws_eks_cluster_endpoint_private_access
  minio_aws_eks_cluster_endpoint_public_access  = var.hello_minio_aws_eks_cluster_endpoint_public_access
  minio_aws_eks_cluster_public_access_cidrs     = var.hello_minio_aws_eks_cluster_public_access_cidrs
  minio_aws_eks_cluster_subnet_ids              = values(module.hello_minio_aws_vpc.minio_aws_subnet_private_ngw_map)
  minio_aws_eks_node_group_name                 = var.hello_minio_aws_eks_node_group_name
  minio_aws_eks_node_group_instance_types       = var.hello_minio_aws_eks_node_group_instance_types
  minio_aws_eks_node_group_desired_size         = var.hello_minio_aws_eks_node_group_desired_size
  minio_aws_eks_node_group_max_size             = var.hello_minio_aws_eks_node_group_max_size
  minio_aws_eks_node_group_min_size             = var.hello_minio_aws_eks_node_group_min_size

}
