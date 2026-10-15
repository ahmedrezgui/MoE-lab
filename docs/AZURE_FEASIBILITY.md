# Azure student GPU feasibility

Checked 2026-10-15. No resources were provisioned and no subscription upgrade was made.

## Account-specific observations

The existing Azure CLI session reports an enabled **Azure for Students** subscription. The subscription policy allows resource deployment in Poland Central, Italy North, UK South, Germany West Central, and Switzerland North. Microsoft.Compute is **NotRegistered**. Compute usage queries returned empty lists; this does not establish usable GPU quota (nor prove every GPU family has zero quota). Current credit balance and approved GPU quota have not been verified.

Microsoft's [subscription limits documentation](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/azure-subscription-service-limits) states that Azure for Students subscriptions are not eligible for quota increases. The [student program documentation](https://learn.microsoft.com/en-us/azure/education-hub/azure-dev-tools-teaching/azure-students-program) describes USD100 credit over 12 months; this is the program allowance, not a verified remaining balance. Credit availability does not guarantee GPU access.

## Concrete price comparison

The public [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices) was queried with:

`armSkuName eq 'Standard_NC4as_T4_v3' and armRegionName eq 'uksouth' and priceType eq 'Consumption'`

Linux results at lookup time:

| Option | USD/hour compute | 10 compute hours |
|---|---:|---:|
| On demand | 0.615 | 6.15 |
| Spot (variable, interruptible) | 0.174414 | 1.74414 |

These retail estimates exclude storage, bandwidth, taxes and any account-specific terms. They are not a deployment quote. Spot price and availability can change. A VM left allocated for 730 hours would be approximately USD448.95 on demand for compute alone; use bounded sessions if cloud training becomes available.

[NCasT4_v3 specifications](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ncast4v3-series) list one NVIDIA T4 with 16 GB VRAM for NC4as_T4_v3. This has four times the local GPU memory, but that does not imply four times the training speed. The T4 also requires fp16 rather than the local bf16 preset; benchmark the same workload before claiming speed benefits.

## Decision

Continue local development on the working RTX 3050 Ti setup. Cloud training is conditional on verified GPU quota, remaining credits or an agreed cash budget, and a benchmark showing a worthwhile benefit for the larger corpus/model. If student GPU quota is unavailable, retain student credits for later small-scale API hosting or artifacts, subject to service eligibility and cost checks. Do not upgrade the subscription simply to run the current tiny demo.

Before provisioning: verify the actual offer/credit balance, register Compute if proceeding, inspect T4-family and total regional vCPU quota in an allowed region, confirm SKU availability, estimate disk/network costs, and prepare a bounded training job with checkpoint persistence and deallocation. Cloud training and portfolio deployment are separate decisions.

Budget constraint confirmed by the user: **student credits only; zero cash spend**. Do not upgrade to Pay-As-You-Go or provision resources with unverified credit eligibility. Local training is the default while GPU entitlement is unresolved.

Final SKU availability check: `az vm list-skus --location uksouth --size Standard_NC4as_T4_v3 --all` returned **NotAvailableForSubscription** for UK South. The same SKU is restricted in West Europe (which is also outside this subscription's allowed-region policy). This confirms the priced UK South T4 option is currently unavailable to this subscription. Other GPU families/allowed regions have not been exhaustively checked. Given the zero-cash constraint, continue locally; the quoted price is a comparison only, not an available student offer.
