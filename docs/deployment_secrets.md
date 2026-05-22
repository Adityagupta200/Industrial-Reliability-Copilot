# GitHub Actions CD Secrets

The CD workflow authenticates to AWS with GitHub OpenID Connect (OIDC). Prefer this over
long-lived `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` secrets.

## Required Repository Secrets

Configure these in GitHub under `Settings -> Secrets and variables -> Actions -> Secrets`.

| Secret | Purpose |
| --- | --- |
| `AWS_ACCOUNT_ID` | Numeric AWS account ID that owns ECR, EKS, and RDS. |
| `AWS_ROLE_TO_ASSUME` | IAM role ARN trusted by the GitHub OIDC provider. |
| `OPENAI_API_KEY` | Runtime OpenAI key injected into Kubernetes as `copilot-secrets`. |
| `POSTGRES_PASSWORD` | RDS application database password. |
| `POSTGRES_DSN` | SQLAlchemy/psycopg DSN for the application database. |
| `INCIDENTS_DB_DSN` | Incident DB DSN used by services that access incident history. |

Example DSNs:

```text
POSTGRES_DSN=postgresql+psycopg://irc:<password>@<rds-endpoint>:5432/industrial_maintenance
INCIDENTS_DB_DSN=postgresql+psycopg://irc:<password>@<rds-endpoint>:5432/industrial_maintenance
```

## Recommended Repository Variables

Configure these in `Settings -> Secrets and variables -> Actions -> Variables`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `AWS_REGION` | `ap-south-1` | AWS region for ECR and EKS. |
| `EKS_CLUSTER_NAME` | `industrial-copilot-cluster` | EKS cluster name used by `aws eks update-kubeconfig`. |

## IAM Role Requirements

`AWS_ROLE_TO_ASSUME` must allow GitHub Actions to assume the role via OIDC. The role needs
least-privilege permissions for:

- ECR auth and image push: `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`,
  `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload`,
  `ecr:PutImage`, and `ecr:DescribeRepositories`.
- EKS kubeconfig access: `eks:DescribeCluster`.
- Kubernetes deploy access through an EKS access entry or `aws-auth` mapping that permits
  the role to update Deployments, Services, ConfigMaps, Secrets, HPAs, PVCs, and Namespaces
  in `staging` and `production`.

Because the CD workflow uses GitHub Environments for `staging` and `production`, the IAM role
trust policy must allow all three OIDC subjects used by the workflow:

```text
repo:Adityagupta200/Industrial-Reliability-Copilot:ref:refs/heads/main
repo:Adityagupta200/Industrial-Reliability-Copilot:environment:staging
repo:Adityagupta200/Industrial-Reliability-Copilot:environment:production
```

Use `infra/aws/github-actions-oidc-trust-policy.template.json` as the source of truth. Replace
`<AWS_ACCOUNT_ID>` with the numeric AWS account ID before updating the IAM role trust policy.

The CD workflow intentionally fails in a `preflight` job if any required secret is missing.
