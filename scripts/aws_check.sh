#!/usr/bin/env bash
# Read-only check that the CHALLENGE AWS account (not HireLens) is configured.
#   bash scripts/aws_check.sh            # uses profile "amlc"
#   AWS_PROFILE=other bash scripts/aws_check.sh
set -uo pipefail
export AWS_PROFILE="${AWS_PROFILE:-amlc}"
export AWS_REGION="${AWS_REGION:-us-east-1}"

if ! aws configure list-profiles | grep -qx "$AWS_PROFILE"; then
  echo "AWS profile '$AWS_PROFILE' not found. Create it with:"
  echo "  aws configure --profile $AWS_PROFILE     # access key of an IAM user in the challenge account, region us-east-1"
  exit 1
fi

echo "== identity ($AWS_PROFILE, $AWS_REGION)"
ARN=$(aws sts get-caller-identity --query Arn --output text) || exit 1
aws sts get-caller-identity --output table
if [[ "$ARN" == *hirelens* ]]; then
  echo "!! This is the HireLens account, not the challenge account. Fix the profile."; exit 1
fi

echo; echo "== SageMaker notebook instances"
aws sagemaker list-notebook-instances \
  --query 'NotebookInstances[].[NotebookInstanceName,NotebookInstanceStatus,InstanceType]' --output table

echo; echo "== SageMaker GPU quotas (0 = request an increase in Service Quotas)"
aws service-quotas list-service-quotas --service-code sagemaker \
  --query "Quotas[?contains(QuotaName,'ml.g4dn.xlarge') || contains(QuotaName,'ml.g5.xlarge') || contains(QuotaName,'ml.t3.medium for notebook')].[QuotaName,Value]" \
  --output table

echo; echo "== running SageMaker endpoints (these bill 24/7 - delete if unused)"
aws sagemaker list-endpoints --query 'Endpoints[].[EndpointName,EndpointStatus]' --output table

echo; echo "== S3 buckets"
aws s3 ls
