output "minio_hostname_private_ips_map" {
  description = "MinIO Hostname Private IPs map"
  value = local.minio_hostname_private_ip_map
}

output "minio_hostname_public_ips_map" {
  description = "MinIO Hostname Public IPs map"
  value = local.minio_hostname_public_ip_map
}

output "hello_minio_instance_unbound" {
  description = "Hello MinIO AWS Unbound IP"
  value = module.hello_minio_instance_unbound.minio_instance_public_ip
}

output "hello_minio_instance_nginx" {
  description = "Hello MinIO AWS Nginx IP"
  value = module.hello_minio_instance_nginx.minio_instance_public_ip
}