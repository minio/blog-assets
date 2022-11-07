variable "minio_aws_vpc_cidr_block" {
  description = "AWS VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "minio_aws_vpc_cidr_newbits" {
  description = "AWS VPC CIDR new bits"
  type        = number
  default     = 4
}

/* */

variable "minio_public_igw_cidr_blocks" {
  type = map(number)
  description = "Availability Zone CIDR Mapping for Public IGW subnets"
 
  default = {
    "us-east-1b" = 1
    "us-east-1d" = 2
    "us-east-1f" = 3
  }
}

variable "minio_private_ngw_cidr_blocks" {
  type = map(number)
  description = "Availability Zone CIDR Mapping for Private NGW subnets"
 
  default = {
    "us-east-1b" = 4
    "us-east-1d" = 5
    "us-east-1f" = 6
  }
}

variable "minio_private_isolated_cidr_blocks" {
  type = map(number)
  description = "Availability Zone CIDR Mapping for Private isolated subnets"
 
  default = {
    "us-east-1b" = 7
    "us-east-1d" = 8
    "us-east-1f" = 9
  }
}