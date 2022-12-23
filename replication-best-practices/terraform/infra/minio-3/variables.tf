variable "hello_minio_region" {
  description = "Hello MinIO AWS region"
  type        = string
}

/* */

variable "hello_minio_cidr_block" {
  description = "Hello MinIO AWS VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "hello_minio_cidr_newbits" {
  description = "Hello MinIO AWS VPC CIDR new bits"
  type        = number
  default     = 4
}

/* */

variable "hello_minio_public_igw_cidr_blocks" {
  type = map(number)
  description = "Hello MinIO Availability Zone CIDR Mapping for Public IGW subnets"
 
  default = {
    "us-east-1b" = 1
    "us-east-1d" = 2
    "us-east-1f" = 3
  }
}

/* */

variable "hello_minio_private_ngw_cidr_blocks" {
  type = map(number)
  description = "Hello MinIO Availability Zone CIDR Mapping for Private NGW subnets"
 
  default = {
    "us-east-1b" = 4
    "us-east-1d" = 5
    "us-east-1f" = 6
  }
}

/* */

variable "hello_minio_private_isolated_cidr_blocks" {
  type = map(number)
  description = "Hello MinIO Availability Zone CIDR Mapping for Private isolated subnets"
 
  default = {
    "us-east-1b" = 7
    "us-east-1d" = 8
    "us-east-1f" = 9
  }
}

/* */

variable "hello_minio_vpc_security_group_rules" {
  description = "Hello World VPC Security Group rules"

  type = list(object({
    rule_type   = string
    description = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))

  default = [
    {
      rule_type   = "ingress"
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "ingress"
      description = "Ping"
      from_port   = 0
      to_port     = 0
      protocol    = "ICMP"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "egress"
      description = "Outbound"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]
}

variable "hello_minio_ami_filter_name_values" {
  description = "Hello World AWS AMI filter name:values"
  type        = list(string)
  default     = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"] 
}

variable "hello_minio_ami_owners" {
  description = "Hello World AWS AMI owners"
  type        = list(string)
  default     = ["099720109477"] 
}

variable "hello_minio_instance_type" {
  description = "Hello World AWS Instance type"
  type        = string
  default     = "t2.micro"
}

variable "hello_minio_instance_key_name" {
  description = "Hello World AWS Instance Key Name"
  type        = string
}

/* Security Groups */

variable "hello_minio_security_group_rules_minio" {
  description = "Hello World Security Group rules MinIO"

  type = list(object({
    rule_type   = string
    description = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))

  default = [
    {
      rule_type   = "ingress"
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
        {
      rule_type   = "ingress"
      description = "MinIO"
      from_port   = 9000
      to_port     = 9000
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "ingress"
      description = "Ping"
      from_port   = 0
      to_port     = 0
      protocol    = "ICMP"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "egress"
      description = "Outbound"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]
}

variable "hello_minio_security_group_rules_nginx" {
  description = "Hello World Security Group rules Nginx"

  type = list(object({
    rule_type   = string
    description = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))

  default = [
    {
      rule_type   = "ingress"
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
        {
      rule_type   = "ingress"
      description = "Nginx"
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "ingress"
      description = "Ping"
      from_port   = 0
      to_port     = 0
      protocol    = "ICMP"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "egress"
      description = "Outbound"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]
}

variable "hello_minio_security_group_rules_unbound" {
  description = "Hello World Security Group rules unbound"

  type = list(object({
    rule_type   = string
    description = string
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))

  default = [
    {
      rule_type   = "ingress"
      description = "SSH"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "ingress"
      description = "DNS TCP"
      from_port   = 53
      to_port     = 53
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "ingress"
      description = "DNS UDP"
      from_port   = 53
      to_port     = 53
      protocol    = "udp"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "ingress"
      description = "Ping"
      from_port   = 0
      to_port     = 0
      protocol    = "ICMP"
      cidr_blocks = ["0.0.0.0/0"]
    },
    {
      rule_type   = "egress"
      description = "Outbound"
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
      cidr_blocks = ["0.0.0.0/0"]
    }
  ]
}
