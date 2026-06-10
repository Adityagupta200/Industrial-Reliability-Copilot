data "aws_caller_identity" "current" {}

locals {
  current_caller_arn             = data.aws_caller_identity.current.arn
  current_caller_is_assumed_role = length(regexall("^arn:[^:]+:sts::[0-9]+:assumed-role/", local.current_caller_arn)) > 0
  current_caller_is_root         = length(regexall("^arn:[^:]+:iam::[0-9]+:root$", local.current_caller_arn)) > 0

  # EKS Access Entries require IAM principal ARNs, not STS session ARNs.
  current_caller_iam_principal_arn = local.current_caller_is_assumed_role ? format(
    "arn:%s:iam::%s:role/%s",
    split(":", local.current_caller_arn)[1],
    data.aws_caller_identity.current.account_id,
    split("/", local.current_caller_arn)[1]
  ) : local.current_caller_arn

  terraform_caller_admin_principal_arns = (
    var.enable_current_caller_cluster_admin && !local.current_caller_is_root
    ? [local.current_caller_iam_principal_arn]
    : []
  )

  eks_admin_principal_arns = distinct(compact(concat(
    local.terraform_caller_admin_principal_arns,
    var.eks_admin_principal_arns
  )))
}

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

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb"                    = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb"           = "1"
  }

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
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.35"

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  authentication_mode = "API_AND_CONFIG_MAP"

  access_entries = merge(
    {
      github_actions_deploy = {
        principal_arn = var.github_actions_role_arn
        type          = "STANDARD"

        policy_associations = {
          cluster_admin = {
            policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
            access_scope = {
              type = "cluster"
            }
          }
        }
      }
    },
    {
      for idx, principal_arn in local.eks_admin_principal_arns :
      "platform_admin_${idx}" => {
        principal_arn = principal_arn
        type          = "STANDARD"

        policy_associations = {
          cluster_admin = {
            policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
            access_scope = {
              type = "cluster"
            }
          }
        }
      }
    }
  )

  eks_managed_node_groups = {
    standard_nodes = {
      # Phase 9 runs staging and production side-by-side and uses maxUnavailable=0
      # rolling updates. The EKS module intentionally ignores desired_size drift,
      # so use a worker size with enough per-node CPU headroom for surge pods.
      min_size       = 3
      max_size       = 6
      desired_size   = 3
      instance_types = ["t3.xlarge"]
      capacity_type  = "ON_DEMAND"
      disk_size      = 50

      ami_type = "AL2023_x86_64_STANDARD"
    }
  }
}

data "aws_iam_policy_document" "ebs_csi_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:kube-system:ebs-csi-controller-sa"]
    }
  }
}

resource "aws_iam_role" "ebs_csi_driver" {
  name               = "${var.cluster_name}-ebs-csi-driver"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume_role.json
}

resource "aws_iam_role_policy_attachment" "ebs_csi_driver" {
  role       = aws_iam_role.ebs_csi_driver.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name             = module.eks.cluster_name
  addon_name               = "aws-ebs-csi-driver"
  service_account_role_arn = aws_iam_role.ebs_csi_driver.arn

  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"

  depends_on = [
    module.eks,
    aws_iam_role_policy_attachment.ebs_csi_driver,
  ]
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

  manage_master_user_password = false

  # Attached explicit security group instead of VPC default
  vpc_security_group_ids = [module.rds_sg.security_group_id]
  create_db_subnet_group = true
  subnet_ids             = module.vpc.private_subnets

  storage_encrypted       = true
  backup_retention_period = 7
  deletion_protection     = true
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

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts_encryption" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts_public_access" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
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

resource "aws_s3_bucket_server_side_encryption_configuration" "documents_encryption" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents_public_access" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# 5. Container Registry (ECR)
resource "aws_ecr_repository" "microservices" {
  #   Removed 'irc-' prefix to match Step 7.5 docker push commands
  for_each = toset([
    "api-gateway",
    "llm-orchestrator",
    "rag-service",
    "anomaly-service"
  ])

  name                 = each.key
  image_tag_mutability = "MUTABLE"
  force_delete         = false

  image_scanning_configuration {
    scan_on_push = true
  }
}
