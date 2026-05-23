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
INCIDENTS_DB_DSN=postgresql+asyncpg://irc:<password>@<rds-endpoint>:5432/industrial_maintenance
```

Get the RDS endpoint from Terraform after `terraform apply`:

```bash
terraform -chdir=infra/terraform output -raw db_instance_endpoint
```

Use the returned RDS hostname in both DSNs. Do not leave placeholders such as
`<rds-endpoint>`, do not use `localhost`, and do not use the in-cluster name `postgres`
for EKS deployments. If Terraform returns an endpoint with `:5432` already appended,
include port `5432` only once in the final DSN.

If the password contains characters such as `@`, `:`, `/`, `#`, or `%`, URL-encode it
before putting it into either DSN.

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

## Local EKS Access

The Terraform stack grants GitHub Actions Kubernetes access through an EKS Access Entry. It also
grants the IAM principal running Terraform cluster-admin access by default through
`enable_current_caller_cluster_admin = true`. This keeps local terminal access managed through
Terraform instead of manual edits to the legacy `aws-auth` ConfigMap.

After `terraform apply`, confirm which local principal was authorized:

```bash
terraform -chdir=infra/terraform output terraform_caller_eks_admin_principal_arn
terraform -chdir=infra/terraform output eks_admin_principal_arns
```

Then refresh kubeconfig and verify authorization:

```bash
aws eks update-kubeconfig --name "$EKS_CLUSTER_NAME" --region "$AWS_REGION"
kubectl auth can-i get pods --all-namespaces
kubectl get pods,deployments,services -n staging
```

If additional engineers or CI/platform roles need local operational access, add them with
`eks_admin_principal_arns`. Find the IAM principal to add:

```bash
aws sts get-caller-identity --query Arn --output text
```

If the ARN is an IAM user, pass it directly:

```text
arn:aws:iam::<account-id>:user/<user-name>
```

If the ARN is an assumed role, convert it from STS form:

```text
arn:aws:sts::<account-id>:assumed-role/<role-name>/<session-name>
```

to IAM role form:

```text
arn:aws:iam::<account-id>:role/<role-name>
```

Then include that ARN when planning/applying Terraform:

```bash
terraform -chdir=infra/terraform plan \
  -var="aws_region=$AWS_REGION" \
  -var="cluster_name=$EKS_CLUSTER_NAME" \
  -var="github_actions_role_arn=$GITHUB_ACTIONS_ROLE_ARN" \
  -var='eks_admin_principal_arns=["arn:aws:iam::<account-id>:user/<user-name>"]' \
  -out=tfplan
```
