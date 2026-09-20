output "bucket_name" {
  description = "S3 bucket for raw data, curated tables and MLflow artifacts."
  value       = aws_s3_bucket.platform.id
}

output "bucket_arn" {
  value = aws_s3_bucket.platform.arn
}

output "github_actions_role_arn" {
  description = "Set this as the AWS_ROLE_ARN repository variable in GitHub."
  value       = aws_iam_role.github_actions.arn
}

output "region" {
  value = data.aws_region.current.region
}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}
