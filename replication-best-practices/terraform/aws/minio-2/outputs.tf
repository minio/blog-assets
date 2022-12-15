output "minio_hostname_ips_map" {
  description = "MinIO Hostname IPs map"
  value = local.minio_hostname_ip_map
}

output "hello_minio_aws_instance_unbound" {
  description = "Hello MinIO AWS Unbound IP"
  value = module.hello_minio_aws_instance_unbound.minio_aws_instance_public_ip
}

output "hello_minio_aws_instance_nginx" {
  description = "Hello MinIO AWS Nginx IP"
  value = module.hello_minio_aws_instance_nginx.minio_aws_instance_public_ip
}