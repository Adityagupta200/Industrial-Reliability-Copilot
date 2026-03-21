output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "db_instance_endpoint" {
  description = "The connection endpoint for the RDS PostgreSQL database"
  value       = module.db.db_instance_endpoint
}