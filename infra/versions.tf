terraform {
  required_version = ">= 1.9"

  required_providers {
    aws = {
      source = "hashicorp/aws"
      # Current major is 6.x. Pessimistic constraint allows 6.x patches and
      # minors but blocks a 7.0 that could break things silently.
      version = "~> 6.0"
    }
  }

  # Local state to start. Once the bucket below exists, uncomment this and run
  # `terraform init -migrate-state` to move state into S3 with native locking.

  backend "s3" {
    bucket       = "mlops-wearables-platform-588804758887"
    key          = "infra/terraform.tfstate"
    region       = "us-east-2"
    encrypt      = true
    use_lockfile = true
  }
}
