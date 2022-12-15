output "minio_aws_security_group_id" {
  description = "MinIO AWS Security Group ID"
  value       = aws_security_group.minio_aws_security_group.id
}
