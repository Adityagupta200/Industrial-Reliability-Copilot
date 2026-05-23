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

variable "dynamodb_lock_table" {
  description = "Name of the DynamoDB table for Terraform state locking"
  type        = string
  default     = "irc-terraform-state-lock"
}

variable "github_actions_role_arn" {
  description = "IAM role ARN assumed by GitHub Actions for CI/CD deployment access to EKS and ECR."
  type        = string
  default     = ""
}

variable "eks_admin_principal_arns" {
  description = "IAM user or role ARNs that should receive cluster-admin access through EKS Access Entries for local operations."
  type        = list(string)
  default     = []
}
