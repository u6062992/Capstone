# Final recommendations to complete EDA

## EDA status

The current EDA has established a reproducible individual donor–fiscal-year
cohort, a $1,200 next-year conversion outcome, three-year historical features,
target-balance diagnostics, feature-quality checks, normalized giving
distributions, pooled lift, fiscal-year lift comparisons, Wilson intervals, and
donor-clustered bootstrap intervals.

The primary cohort contains 154,247 eligible donor-year observations from
FY2023–FY2025, representing 60,372 donors and 101 conversions. Ninety-nine
conversions come from active donors and two come from lapsed donors.

## Main findings to carry forward

1. **The outcome is extremely rare.** The pooled conversion rate is 0.0655%.
   Accuracy is not an appropriate success metric.
2. **Active and lapsed donors are not comparable modeling populations.** The
   active-donor conversion rate is about 0.1004%, while only two lapsed-donor
   conversions are available.
3. **Recent giving level is the strongest simple signal.** Donors currently
   giving $250–$499 have 9.6× pooled lift, and those giving $500–$1,199 have
   108.8× pooled lift.
4. **Multi-year giving adds useful separation.** Three-year giving of $500 or
   more is consistently above baseline, with lift increasing sharply above
   $1,000.
5. **The lift pattern survives donor clustering.** Donor-clustered bootstrap
   intervals remain well above 1× for the actionable high-giving bands.
6. **Cultivation is a treatment, not an ordinary donor attribute.** Contacted
   donors convert much more often, but contact reflects staff selection and may
   also cause conversion.
7. **Passport activity appears incremental rather than dominant.** Viewing
   volume and genre breadth differ modestly; completion percentage does not
   separate converters clearly.
8. **Legacy relationship history may be useful.** Converters have longer
   linked histories, even without assigning household gift amounts to people.

## Required EDA closeout analyses

### 1. Complete the cultivation and holdout analysis

Compare conversion rates among selected donors who were contacted versus
randomly held out, stratified by fiscal year, portfolio rollout, and prior
giving tier. Report donor-clustered or within-year binomial intervals.

This analysis is required before deciding whether cultivation variables belong
in the prediction model. The recommended outputs are:

- A baseline propensity feature set that excludes selection, contact, holdout,
  officer, and portfolio variables
- A separate operational-policy feature set that includes prior cultivation
  history
- A treatment-effect summary using the randomized holdout structure

### 2. Complete Passport missingness and engagement analysis

Treat Passport absence as an observation state rather than zero engagement.
Compare conversion by:

- No observed Passport activity
- One, two, or three active viewing years
- Title-volume bands
- Genre-breadth bands
- Completion-percentage bands among donors with observed viewing

Check these relationships within current-giving tiers. This will determine
whether Passport adds information beyond giving history rather than merely
identifying more engaged existing donors.

### 3. Audit categorical stability and responsible use

Profile conversion and missingness by fiscal year for constituent type, state,
and gender. Use gender and geography primarily for coverage and fairness
auditing, not automatically as model predictors. Combine or suppress very small
categories before reporting rates.

### 4. Remove leakage, constants, and redundant features

Exclude identifiers, target fields, eligibility flags, and constant fields from
model inputs. Review highly redundant pairs, including:

- Current annual giving and threshold gap
- Three-year total and three-year mean giving
- Direct, recurring, nonrecurring, and total giving components
- Campaign counts, responses, and response rates
- Passport observed flags and missing completion measures

Retain raw alternatives in the analytical table, but define a smaller approved
feature list before modeling.

### 5. Assess temporal stability and synthetic-data mechanisms

Compare distributions and conversion rates across FY2023, FY2024, and FY2025.
Document features whose meaning or distribution changes materially. The sharp
giving-tier relationship may reflect the synthetic data-generation process, so
the final report should distinguish patterns useful for demonstrating the
workflow from claims expected to generalize to production PBS Utah data.

### 6. Establish simple operational baselines

Before fitting a predictive model, calculate precision, recall, conversion
capture, and lift for transparent rules such as:

- Current annual giving of at least $250
- Current annual giving of at least $500
- Three-year giving of at least $500
- Three-year giving of at least $1,000
- Ranking by current annual giving
- Ranking by three-year total giving

Evaluate these rules at PBS Utah's documented annual portfolio capacity of
approximately 1,000 prospects. A model should outperform these baselines at the
same top-K capacity.

### 7. Finalize EDA documentation

Create a feature dictionary for the enriched table and a final EDA report that
records:

- Cohort and outcome definitions
- Exclusion and deduplication rules
- Missingness interpretations
- Feature availability dates
- Leakage exclusions
- Active-versus-lapsed decision
- Cultivation treatment handling
- Visualizations and uncertainty intervals
- Limitations of synthetic and repeated donor-year data

## Recommended modeling handoff

### Primary population

Use active normal donors for the first propensity model. Retain lapsed donors as
a separately reported population until more positive outcomes or a different
reactivation target is available.

### Feature sets

1. **Baseline propensity:** giving, campaign, Passport, constituent, and legacy
   history known at the scoring date; exclude cultivation treatment variables.
2. **Operational-policy model:** add prior cultivation and portfolio history to
   predict outcomes under the existing outreach process.
3. **Treatment analysis:** use contacted-versus-holdout records separately to
   estimate incremental cultivation effects.

### Validation design

Use time-based evaluation rather than random row splitting. A reasonable first
design is to develop with FY2023–FY2024 and hold out FY2025 as the final temporal
test. Keep donor identifiers out of the model and report clustered uncertainty
where pooled donor-year estimates are presented.

### Success metrics

Prioritize precision at K, recall at K, lift at K, precision-recall AUC, and
probability calibration. Report results at K = 1,000 and at smaller operating
capacities agreed with PBS Utah. Do not use overall accuracy as the primary
metric.

## EDA completion criterion

EDA can be considered complete when the cultivation/holdout, Passport
incremental-value, categorical stability, redundancy, temporal stability, and
top-K baseline analyses are documented and an approved feature list is frozen.
At that point, the project can proceed to model development without changing
the cohort or target definition.
