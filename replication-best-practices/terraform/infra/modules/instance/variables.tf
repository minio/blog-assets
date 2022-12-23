variable "minio_subnet_id" {
  description = "MinIO AWS Subnet ID"
  type        = string
}

variable "minio_vpc_security_group_ids" {
  description = "MinIO VPC Security Group IDs"
  type        = list(string)
}

variable "minio_ami_filter_name_values" {
  description = "MinIO AWS AMI filter name:values"
  type        = list(string)
  default     = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"] 
}

variable "minio_ami_owners" {
  description = "MinIO AWS AMI owners"
  type        = list(string)
  default     = ["099720109477"] 
}

variable "minio_instance_type" {
  description = "MinIO AWS Instance type"
  type        = string
  default     = "t2.micro"
}

variable "minio_instance_key_name" {
  description = "MinIO AWS Instance Key Name"
  type        = string
}

variable "minio_instance_user_data" {
  description = "MinIO AWS Instance user data"
  type        = string
  default     = ""
}

variable "minio_instance_user_data_replace_on_change" {
  description = "MinIO AWS Instance user data replace on change"
  type        = bool
  default     = false
}
