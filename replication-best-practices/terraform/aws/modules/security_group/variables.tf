variable "minio_aws_security_group_vpc_id" {
  description = "MinIO AWS VPC ID"
  type        = string
}

variable "minio_aws_security_group_rules" {
  description = "MinIO VPC Security Group rules"

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
