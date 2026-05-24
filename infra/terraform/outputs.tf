output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "db_instance_endpoint" {
  description = "The connection endpoint for the RDS PostgreSQL database"
  value       = module.db.db_instance_endpoint
}

output "ecr_repository_urls" {
  description = "The URIs for the deployed ECR repositories"
  value       = { for k, v in aws_ecr_repository.microservices : k => v.repository_url }
}

output "eks_admin_principal_arns" {
  description = "IAM principals granted Kubernetes cluster-admin access through EKS Access Entries."
  value       = local.eks_admin_principal_arns
}

output "terraform_caller_eks_admin_principal_arn" {
  description = "Normalized IAM principal ARN for the identity that ran Terraform and was granted EKS cluster-admin access."
  value       = local.current_caller_is_root ? null : local.current_caller_iam_principal_arn
}

output "github_actions_eks_admin_principal_arn" {
  description = "GitHub Actions IAM role granted Kubernetes cluster-admin access through an EKS Access Entry."
  value       = var.github_actions_role_arn
}
