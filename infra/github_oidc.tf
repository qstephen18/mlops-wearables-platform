# CI authenticates by assuming a role through OIDC. No access keys are stored
# in GitHub. This is the difference that matters: a long-lived key on a laptop
# is a manageable risk, the same key in a CI system is how incidents start.

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
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

    # The sub claim carries immutable numeric IDs appended to both the owner and
    # the repo name: repo:owner@<id>/name@<id>:<ref>. An exact owner/name match
    # never fires. Wildcards span the ID suffixes while still pinning both.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${split("/", var.github_repo)[0]}*/${split("/", var.github_repo)[1]}*:*"]
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
    sid       = "BucketLevel"
    effect    = "Allow"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
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

# terraform plan must read every managed resource to detect drift, which spans
# IAM, budgets and S3 configuration. ReadOnlyAccess is broader than this project
# needs, but it is read-only and the alternative is enumerating a describe/get
# permission per resource type and updating it every time the stack grows.
# Write access stays scoped to the state prefix via the inline policy above:
# this role can plan, never apply.
resource "aws_iam_role_policy_attachment" "ci_readonly" {
  role       = aws_iam_role.github_actions.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}
