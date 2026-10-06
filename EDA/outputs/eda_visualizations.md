# Initial EDA visualizations

## Target balance

![Target balance and conversion rates](figures/target_balance.svg)

## Giving distributions

![Giving distributions by conversion outcome](figures/giving_distributions.svg)

## Converter lift

![Converter lift by giving band](figures/converter_lift.svg)

Error bars are 95% Wilson intervals. Lift intervals divide the Wilson rate
interval by the overall cohort conversion rate; uncertainty in the baseline
rate is not propagated.

## Fiscal-year comparison

![Conversion rates by fiscal year and giving tier](figures/converter_lift_by_fiscal_year.svg)

## Donor-clustered bootstrap intervals

![Donor-clustered bootstrap lift intervals](figures/converter_lift_cluster_bootstrap.svg)

These intervals resample complete donor histories so repeated donor-year
observations remain together.

The underlying lift values are available in
[`converter_lift_by_giving_band.csv`](converter_lift_by_giving_band.csv) and
[`converter_lift_by_fiscal_year.csv`](converter_lift_by_fiscal_year.csv).
Clustered-bootstrap intervals are available in
[`converter_lift_cluster_bootstrap.csv`](converter_lift_cluster_bootstrap.csv).
