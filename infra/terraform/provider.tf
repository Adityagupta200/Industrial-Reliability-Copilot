terraform {
  required_version = ">= 1.5.0"
  
  # Step 7.4 Requirement: S3 Backend and DynamoDB State Locking
  backend "s3" {
    bucket         = "irc-terraform-state-bucket"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "irc-terraform-state-locks"
    encrypt        = true
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