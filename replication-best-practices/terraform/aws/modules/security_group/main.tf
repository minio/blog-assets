resource "aws_security_group" "minio_aws_security_group" {
 
  vpc_id = var.minio_aws_security_group_vpc_id

}

resource "aws_security_group_rule" "minio_aws_security_group_rule" {
  for_each = {for sg in var.minio_aws_security_group_rules:  sg.description => sg}

  type        = each.value.rule_type
  description = each.value.description
  from_port   = each.value.from_port
  to_port     = each.value.to_port
  protocol    = each.value.protocol
  cidr_blocks = each.value.cidr_blocks
 
  security_group_id = aws_security_group.minio_aws_security_group.id
}
