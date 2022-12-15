output "minio_aws_instance_id" {
  description = "MinIO AWS Instance ID"
  value       = aws_instance.minio_aws_instance.id
}

output "minio_aws_instance_public_ip" {
  description = "MinIO AWS Instance Public IP"
  value       = aws_instance.minio_aws_instance.public_ip
}

output "minio_aws_instance_private_ip" {
  description = "MinIO AWS Instance Private IP"
  value       = aws_instance.minio_aws_instance.private_ip
}
