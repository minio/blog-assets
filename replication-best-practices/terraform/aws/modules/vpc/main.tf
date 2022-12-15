resource "aws_vpc" "minio_aws_vpc" {

  cidr_block           = var.minio_aws_vpc_cidr_block
  instance_tenancy     = "default"
  enable_dns_hostnames = true

}

/* */

resource "aws_subnet" "minio_aws_subnet_public_igw" {

  for_each = var.minio_public_igw_cidr_blocks

  vpc_id            = aws_vpc.minio_aws_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.minio_aws_vpc.cidr_block, var.minio_aws_vpc_cidr_newbits, each.value)
  availability_zone = each.key

  map_public_ip_on_launch = true
}

resource "aws_route_table" "minio_aws_route_table_public_igw" {

  vpc_id = aws_vpc.minio_aws_vpc.id

}

resource "aws_route_table_association" "minio_aws_route_table_association_public_igw" {

  for_each       = aws_subnet.minio_aws_subnet_public_igw

  subnet_id      = each.value.id
  route_table_id = aws_route_table.minio_aws_route_table_public_igw.id
}

resource "aws_internet_gateway" "minio_aws_internet_gateway" {

  vpc_id = aws_vpc.minio_aws_vpc.id

}

resource "aws_route" "minio_aws_route_public_igw" {
  route_table_id         = aws_route_table.minio_aws_route_table_public_igw.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.minio_aws_internet_gateway.id
}

/* */

resource "aws_subnet" "minio_aws_subnet_private_ngw" {

  for_each = var.minio_private_ngw_cidr_blocks

  vpc_id            = aws_vpc.minio_aws_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.minio_aws_vpc.cidr_block, var.minio_aws_vpc_cidr_newbits, each.value)
  availability_zone = each.key
}

resource "aws_route_table" "minio_aws_route_table_private_ngw" {

  for_each = var.minio_private_ngw_cidr_blocks

  vpc_id = aws_vpc.minio_aws_vpc.id
}

resource "aws_route_table_association" "minio_aws_route_table_association_private_ngw" {

  for_each = var.minio_private_ngw_cidr_blocks

  subnet_id      = aws_subnet.minio_aws_subnet_private_ngw[each.key].id
  route_table_id = aws_route_table.minio_aws_route_table_private_ngw[each.key].id
}

resource "aws_eip" "minio_aws_eip_nat" {

  for_each = var.minio_private_ngw_cidr_blocks

  vpc = true
}

resource "aws_nat_gateway" "minio_aws_nat_gateway" {

  for_each = var.minio_private_ngw_cidr_blocks

  subnet_id     = aws_subnet.minio_aws_subnet_public_igw[each.key].id
  allocation_id = aws_eip.minio_aws_eip_nat[each.key].id

  depends_on    = [aws_internet_gateway.minio_aws_internet_gateway]
}

resource "aws_route" "minio_aws_route_private_ngw" {

  for_each = var.minio_private_ngw_cidr_blocks

  route_table_id         = aws_route_table.minio_aws_route_table_private_ngw[each.key].id
  destination_cidr_block = "0.0.0.0/0"
  nat_gateway_id         = aws_nat_gateway.minio_aws_nat_gateway[each.key].id
}

/* */

resource "aws_subnet" "minio_aws_subnet_private_isolated" {

  for_each = var.minio_private_isolated_cidr_blocks

  vpc_id            = aws_vpc.minio_aws_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.minio_aws_vpc.cidr_block, var.minio_aws_vpc_cidr_newbits, each.value)
  availability_zone = each.key
}

resource "aws_route_table" "minio_aws_route_table_private_isolated" {

  vpc_id = aws_vpc.minio_aws_vpc.id

}

resource "aws_route_table_association" "minio_aws_route_table_association_private_isolated" {

  for_each = aws_subnet.minio_aws_subnet_private_isolated

  subnet_id      = each.value.id
  route_table_id = aws_route_table.minio_aws_route_table_private_isolated.id
}