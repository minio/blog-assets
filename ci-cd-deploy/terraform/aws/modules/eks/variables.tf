variable "minio_aws_eks_cluster_subnet_ids" {
  description = "AWS EKS Cluster subnet IDs"
  type        = list(string)
}

variable "minio_aws_eks_cluster_name" {
  description = "AWS EKS Cluster name"
  type        = string
  default     = "minio_aws_eks_cluster"
}

variable "minio_aws_eks_cluster_endpoint_private_access" {
  description = "AWS EKS Cluster endpoint private access"
  type        = bool
  default     = true
}

variable "minio_aws_eks_cluster_endpoint_public_access" {
  description = "AWS EKS Cluster endpoint public access"
  type        = bool
  default     = true
}

variable "minio_aws_eks_cluster_public_access_cidrs" {
  description = "AWS EKS Cluster public access cidrs"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "minio_aws_eks_node_group_name" {
  description = "AWS EKS Node group name"
  type        = string
  default     = "minio_aws_eks_node_group"
}

variable "minio_aws_eks_node_group_instance_types" {
  description = "AWS EKS Node group instance types"
  type        = list(string)
  default     = ["t3.large"]
}

variable "minio_aws_eks_node_group_desired_size" {
  description = "AWS EKS Node group desired size"
  type        = number
  default     = 3
}

variable "minio_aws_eks_node_group_max_size" {
  description = "AWS EKS Node group max size"
  type        = number
  default     = 5
}

variable "minio_aws_eks_node_group_min_size" {
  description = "AWS EKS Node group min size"
  type        = number
  default     = 1
}