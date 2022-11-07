output "minio_aws_vpc_id" {
  description = "AWS VPC ID"
  value       = aws_vpc.minio_aws_vpc.id
}

output "minio_aws_subnet_public_igw_map" {
  description = "AWS Subnet Public IGW ID and block"
  value = {
    for subnet in aws_subnet.minio_aws_subnet_public_igw :
    subnet.availability_zone => subnet.id
  }
}

output "minio_aws_subnet_private_ngw_map" {
  description = "AWS Subnet Private NGW ID and block"
  value = {
    for subnet in aws_subnet.minio_aws_subnet_private_ngw :
    subnet.availability_zone => subnet.id
  }
}

output "minio_aws_subnet_private_isolated_map" {
  description = "AWS Subnet Private isolated ID and block"
  value = {
    for subnet in aws_subnet.minio_aws_subnet_private_isolated :
    subnet.availability_zone => subnet.id
  }
}