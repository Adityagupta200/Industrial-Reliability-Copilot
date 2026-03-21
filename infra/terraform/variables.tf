variable "aws_region" {
  type    = string
  default = "ap-south-1"
}

variable "cluster_name" {
  type    = string
  default = "industrial-copilot-cluster"
}

variable "db_password" {
  description = "Password for PostgreSQL RDS database"
  type        = string
  sensitive   = true
}