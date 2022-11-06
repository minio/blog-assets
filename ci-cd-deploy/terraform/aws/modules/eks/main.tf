resource "aws_iam_role" "minio_aws_iam_role_eks_cluster" {
  name = "minio_aws_iam_role_eks_cluster"
  path = "/"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

}

resource "aws_iam_role_policy_attachment" "AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role    = aws_iam_role.minio_aws_iam_role_eks_cluster.name
}
resource "aws_iam_role_policy_attachment" "AmazonEC2ContainerRegistryReadOnly-EKS" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role    = aws_iam_role.minio_aws_iam_role_eks_cluster.name
}

resource "aws_eks_cluster" "minio_aws_eks_cluster" {
  name = var.minio_aws_eks_cluster_name
  role_arn = aws_iam_role.minio_aws_iam_role_eks_cluster.arn

  vpc_config {
    subnet_ids              = var.minio_aws_eks_cluster_subnet_ids
    endpoint_private_access = var.minio_aws_eks_cluster_endpoint_private_access
    endpoint_public_access  = var.minio_aws_eks_cluster_endpoint_public_access
    public_access_cidrs     = var.minio_aws_eks_cluster_public_access_cidrs
  }

  depends_on = [
    aws_iam_role.minio_aws_iam_role_eks_cluster,
  ]

}

resource "aws_iam_role" "minio_aws_iam_role_eks_worker" {
  name = "minio_aws_iam_role_eks_worker"

  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
 
}
 
resource "aws_iam_role_policy_attachment" "AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role    = aws_iam_role.minio_aws_iam_role_eks_worker.name
}
 
resource "aws_iam_role_policy_attachment" "AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role    = aws_iam_role.minio_aws_iam_role_eks_worker.name
}
 
resource "aws_iam_role_policy_attachment" "AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role    = aws_iam_role.minio_aws_iam_role_eks_worker.name
}

resource "aws_eks_node_group" "minio_aws_eks_node_group" {
  cluster_name    = aws_eks_cluster.minio_aws_eks_cluster.name
  node_group_name = var.minio_aws_eks_node_group_name
  node_role_arn   = aws_iam_role.minio_aws_iam_role_eks_worker.arn
  subnet_ids      = var.minio_aws_eks_cluster_subnet_ids
  instance_types  = var.minio_aws_eks_node_group_instance_types
 
  scaling_config {
    desired_size = var.minio_aws_eks_node_group_desired_size
    max_size     = var.minio_aws_eks_node_group_max_size
    min_size     = var.minio_aws_eks_node_group_min_size
  }

  depends_on = [
    aws_iam_role.minio_aws_iam_role_eks_worker,
  ]
 
}