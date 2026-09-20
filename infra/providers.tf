provider "aws" {
  region = var.region

  # Every resource this project creates carries these tags automatically.
  # This is what makes the cost story real: spend can be sliced by Project
  # without anyone remembering to tag a resource by hand.
  default_tags {
    tags = {
      Project     = var.project
      Owner       = var.owner
      ManagedBy   = "terraform"
      Environment = var.environment
      CostCenter  = "personal-lab"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}
