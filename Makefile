.PHONY: help init fmt validate plan apply destroy cost data
.DEFAULT_GOAL := help

TF := terraform -chdir=infra

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

init: ## Initialize Terraform
	$(TF) init

fmt: ## Format Terraform files
	$(TF) fmt -recursive

validate: fmt ## Validate configuration
	$(TF) validate

plan: validate ## Show what would change
	$(TF) plan

apply: ## Apply changes
	$(TF) apply

destroy: ## Tear down infra. WARNING: state lives in this bucket — see README.
	$(TF) destroy

cost: ## Month-to-date spend. Run when you sit down and when you stop.
	@aws ce get-cost-and-usage \
		--time-period Start=$$(date -u +%Y-%m-01),End=$$(date -u -v+1d +%Y-%m-%d 2>/dev/null || date -u -d '+1 day' +%Y-%m-%d) \
		--granularity MONTHLY \
		--metrics UnblendedCost \
		--query 'ResultsByTime[0].Total.UnblendedCost' \
		--output table

data: ## Download PAMAP2 into ./data (gitignored)
	@bash scripts/download_data.sh
