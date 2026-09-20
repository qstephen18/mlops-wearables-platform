variable "region" {
  description = "AWS region. Everything stays in one region to avoid inter-region transfer charges."
  type        = string
  default     = "us-east-2"
}

variable "project" {
  description = "Project name. Used for resource naming and cost allocation tags."
  type        = string
  default     = "mlops-wearables-platform"
}

variable "owner" {
  description = "Resource owner tag."
  type        = string
  default     = "qstephen18"
}

variable "environment" {
  description = "Environment name for tagging."
  type        = string
  default     = "lab"
}

variable "github_repo" {
  description = "GitHub repo allowed to assume the CI role, as owner/name."
  type        = string
  default     = "qstephen18/mlops-wearables-platform"
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget in USD. Alerts fire against this, they do not cap spend."
  type        = number
  default     = 10
}

variable "budget_alert_email" {
  description = "Email address for budget notifications."
  type        = string
  # No default on purpose: set this in terraform.tfvars, which is gitignored.
}
