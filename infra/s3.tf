# Single bucket, prefix-separated. Account ID suffix because S3 bucket names
# are globally unique across all of AWS, not just your account.
locals {
  bucket_name = "${var.project}-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "platform" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_versioning" "platform" {
  bucket = aws_s3_bucket.platform.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "platform" {
  bucket = aws_s3_bucket.platform.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "platform" {
  bucket = aws_s3_bucket.platform.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Versioning is on for reproducibility, but old versions are storage you pay
# for forever. 30 days is long enough to recover a mistake, short enough that
# the bill stays flat.
resource "aws_s3_bucket_lifecycle_configuration" "platform" {
  bucket     = aws_s3_bucket.platform.id
  depends_on = [aws_s3_bucket_versioning.platform]

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
