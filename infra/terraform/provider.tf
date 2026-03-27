terraform {
  required_version = ">= 1.5.0"

  # Step 7.4 Requirement: S3 Backend with Native S3 State Locking (Modern Standard)
  backend "s3" {
    # CRITICAL: Update '12345' to match the exact globally unique bucket name you created via the AWS CLI
    bucket = "irc-terraform-state-bucket-12345"

    key     = "production/terraform.tfstate"
    region  = "us-east-1"
    encrypt = true

    # Modern replacement for the deprecated dynamodb_table locking
    use_lockfile = true
  }

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