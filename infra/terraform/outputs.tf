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