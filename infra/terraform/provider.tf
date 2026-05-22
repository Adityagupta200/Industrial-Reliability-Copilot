terraform {
  required_version = ">= 1.5.0"

  # Backend values are supplied by `terraform init -backend-config=...`.
  # This avoids committing account-specific state bucket names to source control.
  backend "s3" {}

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
