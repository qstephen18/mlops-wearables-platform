# CI authenticates by assuming a role through OIDC. No access keys are stored
# in GitHub. This is the difference that matters: a long-lived key on a laptop
# is a manageable risk, the same key in a CI system is how incidents start.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  # AWS no longer validates thumbprints for well-known IdPs, but the API still
  # requires the field. This value is GitHub's published root and is inert.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Scoped to this repository only. Without this condition, any GitHub repo
    # anywhere could assume the role.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${var.project}-github-actions"
  description        = "Assumed by GitHub Actions in ${var.github_repo} via OIDC"
  assume_role_policy = data.aws_iam_policy_document.github_assume_role.json
}

data "aws_iam_policy_document" "ci_permissions" {
  statement {
    sid     = "BucketLevel"
    effect  = "Allow"
    actions = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.platform.arn]
  }

  statement {
    sid    = "ObjectLevel"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${aws_s3_bucket.platform.arn}/*"]
  }
}

resource "aws_iam_role_policy" "ci_permissions" {
  name   = "${var.project}-ci"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.ci_permissions.json
}
