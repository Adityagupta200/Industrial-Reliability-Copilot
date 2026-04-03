terraform {
  required_version = ">= 1.5.0"

  # Step 7.4 Requirement: S3 Backend with DynamoDB State Locking
  backend "s3" {
    # CRITICAL: Update '12345' to match the exact globally unique bucket name you created
    bucket         = "irc-terraform-state-bucket-12345"
    key            = "production/terraform.tfstate"
    region         = "us-east-1" # Or ap-south-1 depending on your bucket
    encrypt        = true

    # FIX: Strictly adhering to PDF spec for DynamoDB locking
    dynamodb_table = "irc-terraform-state-lock"
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