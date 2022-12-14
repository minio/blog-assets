data "aws_ami" "minio_ami" {
  most_recent = true

  filter {
    name   = "name"
    values = var.minio_ami_filter_name_values
  }

  owners = var.minio_ami_owners
}

resource "aws_instance" "minio_instance" {

  ami                         = data.aws_ami.minio_ami.id
  instance_type               = var.minio_instance_type
  subnet_id                   = var.minio_subnet_id
  vpc_security_group_ids      = var.minio_vpc_security_group_ids
  key_name                    = var.minio_instance_key_name
  user_data                   = var.minio_instance_user_data
  user_data_replace_on_change = var.minio_instance_user_data_replace_on_change

}
