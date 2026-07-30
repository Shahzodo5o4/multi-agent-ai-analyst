# Q3 2024 Churn Postmortem

**Status:** Final · **Owner:** Customer Success · **Reviewed:** 12 October 2024

## Summary

Churn in Q3 2024 was materially above trend. The dominant driver was the
pricing change announced on 1 July 2024, which moved the Starter plan from a
flat $29/month to $49/month and removed the two free seats that SMB accounts
had previously relied on.

## What happened

On 1 July 2024 we published new pricing with 14 days' notice. Three decisions
compounded each other:

1. **Notice was too short.** Most SMB customers are billed monthly and had no
   time to budget for the increase before their next invoice.
2. **The free seats were removed silently.** The change was in the pricing page
   footnotes but was not in the announcement email. Teams of four discovered the
   change only when they were charged for two extra seats.
3. **No grandfathering.** Competitors who ran comparable increases in 2023
   grandfathered existing customers for 12 months. We did not.

## Contributing factors

- **Missing integrations.** A secondary cluster of churn came from accounts
  waiting on the Salesforce and HubSpot connectors, both of which slipped from
  Q2 to Q4. These customers were already at risk; the price increase converted
  hesitation into cancellation.
- **Onboarding gaps.** Accounts that never completed onboarding churned at
  roughly three times the rate of those that did. The pricing change hit this
  group hardest because they had not yet seen value.
- **Support backlog.** Billing ticket volume tripled in July. Median first
  response time went from 4 hours to 31 hours, so customers who wanted to
  negotiate simply cancelled instead.

## Segment breakdown

Churn was concentrated in **SMB accounts on the Starter plan**. Mid-Market and
Enterprise accounts, which are on annual contracts, were largely insulated —
their renewals do not come up until 2025, so the impact there is deferred rather
than avoided.

## Actions taken

| Action | Owner | Status |
|---|---|---|
| Reinstate two free seats for accounts created before July 2024 | Billing | Done, 20 Aug 2024 |
| 90-day price freeze for any account that requests one | Success | Done, 1 Sep 2024 |
| Ship Salesforce connector | Platform | Shipped Q4 2024 |
| Minimum 60 days' notice for any future pricing change | Finance | Policy updated |
| Staff billing support to a 8-hour first response SLA | Support | Done |

## Lessons

Price increases are survivable; **surprise** price increases are not. Every
future change must ship with the notice period, the grandfathering terms, and
the affected-seat count stated in the announcement itself — not in a footnote.
