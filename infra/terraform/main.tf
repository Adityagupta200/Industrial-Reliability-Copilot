data "aws_caller_identity" "current" {}

# 1. Networking (VPC)
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.1.2"

  name = "irc-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["ap-south-1a", "ap-south-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = {
    Environment = "production"
  }
}

# Phase 7 Production Security: Dedicated Security Group for RDS
module "rds_sg" {
  source  = "terraform-aws-modules/security-group/aws"
  version = "~> 5.1.0"

  name        = "irc-rds-sg"
  description = "Security group for RDS PostgreSQL allowing internal VPC traffic"
  vpc_id      = module.vpc.vpc_id

  ingress_with_cidr_blocks = [
    {
      from_port   = 5432
      to_port     = 5432
      protocol    = "tcp"
      description = "PostgreSQL access from EKS Subnets"
      cidr_blocks = "10.0.0.0/16"
    }
  ]
}

# 2. Kubernetes Cluster (EKS)
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "19.17.2"

  cluster_name    = var.cluster_name
  cluster_version = "1.31"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    standard_nodes = {
      min_size       = 2
      max_size       = 10
      desired_size   = 2
      instance_types = ["t3.medium"] # Matches PDF spec

      # AL2023 explicitly defined to pass Step 7.4 verification
      ami_type = "AL2023_x86_64_STANDARD"
    }
  }
}

# 3. PostgreSQL Database (RDS)
module "db" {
  source  = "terraform-aws-modules/rds/aws"
  version = "6.1.1"

  identifier           = "irc-postgres-db"
  engine               = "postgres"
  engine_version       = "15"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = "db.t3.micro"
  allocated_storage    = 20

  db_name  = "industrial_maintenance"
  username = "irc"
  password = var.db_password

  # FIX: Attached explicit security group instead of VPC default
  vpc_security_group_ids = [module.rds_sg.security_group_id]
  create_db_subnet_group = true
  subnet_ids             = module.vpc.private_subnets

  # FIX: Phase 7 mandatory production flags
  storage_encrypted       = true
  backup_retention_period = 7
  skip_final_snapshot     = false # True is for dev only
}

# 4. Storage (S3) for Models & Documents
resource "aws_s3_bucket" "artifacts" {
  bucket = "irc-artifacts-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "artifacts_versioning" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket" "documents" {
  bucket = "irc-documents-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_versioning" "documents_versioning" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

# 5. Container Registry (ECR)
resource "aws_ecr_repository" "microservices" {
  # PRODUCTION FIX: Removed 'irc-' prefix to match Step 7.5 docker push commands
  for_each = toset([
    "api-gateway",
    "llm-orchestrator",
    "rag-service",
    "anomaly-service"
  ])

  name                 = each.key
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}