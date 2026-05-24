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

  validation {
    condition     = can(regex("^arn:[^:]+:iam::[0-9]{12}:role/.+", var.github_actions_role_arn))
    error_message = "github_actions_role_arn must be the IAM role ARN used by the GitHub AWS_ROLE_TO_ASSUME secret, for example arn:aws:iam::<account-id>:role/GitHubActionsIndustrialCopilotDeploy. Do not use an STS assumed-role ARN."
  }
}

variable "eks_admin_principal_arns" {
  description = "IAM user or role ARNs that should receive cluster-admin access through EKS Access Entries for local operations."
  type        = list(string)
  default     = []
}

variable "enable_current_caller_cluster_admin" {
  description = "Grant the IAM principal running Terraform cluster-admin access through an EKS Access Entry. Disable only when platform access is managed outside this stack."
  type        = bool
  default     = true
}
