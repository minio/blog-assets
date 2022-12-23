terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.31.0"
    }
  }

  backend "s3" {
    bucket = "aj-terraform-bucket"
    key    = "tf/aj/mo2"
    region = "us-east-1"
  }

}

provider "aws" {
  region  = var.hello_minio_region
}

module "hello_minio_vpc" {
  source = "../modules/vpc"

  minio_cidr_block   = var.hello_minio_cidr_block
  minio_cidr_newbits = var.hello_minio_cidr_newbits

  minio_public_igw_cidr_blocks       = var.hello_minio_public_igw_cidr_blocks
  minio_private_ngw_cidr_blocks      = var.hello_minio_private_ngw_cidr_blocks
  minio_private_isolated_cidr_blocks = var.hello_minio_private_isolated_cidr_blocks

}

/* EC2 Example */

locals {
  common_public_subnet = element(keys(var.hello_minio_public_igw_cidr_blocks), 0)
  common_private_subnet = element(keys(var.hello_minio_private_ngw_cidr_blocks), 0)
}

module "hello_minio_security_group_unbound" {
  source = "../modules/security_group"

  minio_security_group_vpc_id = module.hello_minio_vpc.minio_vpc_id
  minio_security_group_rules  = var.hello_minio_security_group_rules_unbound

}

module "hello_minio_instance_unbound" {
  source = "../modules/instance"

  minio_vpc_security_group_ids = [module.hello_minio_security_group_unbound.minio_security_group_id]
  minio_subnet_id          = module.hello_minio_vpc.minio_subnet_public_igw_map[local.common_public_subnet]

  minio_ami_filter_name_values = var.hello_minio_ami_filter_name_values
  minio_ami_owners             = var.hello_minio_ami_owners
  minio_instance_type          = var.hello_minio_instance_type
  minio_instance_key_name      = var.hello_minio_instance_key_name

  minio_instance_user_data     = templatefile("../templates/_user_data_unbound.sh.tftpl", {})

}

module "hello_minio_security_group_minio" {
  source = "../modules/security_group"

  minio_security_group_vpc_id = module.hello_minio_vpc.minio_vpc_id
  minio_security_group_rules  = var.hello_minio_security_group_rules_minio

}

module "hello_minio_instance_minio" {
  source = "../modules/instance"

  for_each = var.hello_minio_private_ngw_cidr_blocks

  minio_vpc_security_group_ids = [module.hello_minio_security_group_minio.minio_security_group_id]
  minio_subnet_id          = module.hello_minio_vpc.minio_subnet_private_ngw_map[each.key]

  minio_ami_filter_name_values = var.hello_minio_ami_filter_name_values
  minio_ami_owners             = var.hello_minio_ami_owners
  minio_instance_type          = var.hello_minio_instance_type
  minio_instance_key_name      = var.hello_minio_instance_key_name

  minio_instance_user_data     = templatefile("../templates/_user_data_minio.sh.tftpl", {
    minio_version = "20221207005637.0.0"
    unbound_ip = module.hello_minio_instance_unbound.minio_instance_private_ip
  })

}

resource "aws_ebs_volume" "ebs_volume_minio" {

  for_each = var.hello_minio_private_ngw_cidr_blocks

  availability_zone = each.key

  size = 10
 
}

resource "aws_volume_attachment" "volume_attachment_minio" {

    for_each = var.hello_minio_private_ngw_cidr_blocks

    device_name = "/dev/xvdb"
    volume_id = aws_ebs_volume.ebs_volume_minio[each.key].id
    instance_id = module.hello_minio_instance_minio[each.key].minio_instance_id

}

locals {

      minio_instances_ips = [
        for each_instance in module.hello_minio_instance_minio :
        each_instance.minio_instance_private_ip
      ]

      minio_hostname_ip_map = {
        for i, ip in local.minio_instances_ips :
          "server-${i+1}.minio.local" => ip
    }

}

module "hello_minio_security_group_nginx" {
  source = "../modules/security_group"

  minio_security_group_vpc_id = module.hello_minio_vpc.minio_vpc_id
  minio_security_group_rules  = var.hello_minio_security_group_rules_nginx

}

module "hello_minio_instance_nginx" {
  source = "../modules/instance"

  minio_vpc_security_group_ids = [module.hello_minio_security_group_nginx.minio_security_group_id]
  minio_subnet_id          = module.hello_minio_vpc.minio_subnet_public_igw_map[local.common_public_subnet]

  minio_ami_filter_name_values = var.hello_minio_ami_filter_name_values
  minio_ami_owners             = var.hello_minio_ami_owners
  minio_instance_type          = var.hello_minio_instance_type
  minio_instance_key_name      = var.hello_minio_instance_key_name

  minio_instance_user_data     = templatefile("../templates/_user_data_nginx.sh.tftpl", {
    unbound_ip = module.hello_minio_instance_unbound.minio_instance_private_ip
  })

  minio_instance_user_data_replace_on_change = false


  depends_on = [
    module.hello_minio_instance_unbound
  ]

}
