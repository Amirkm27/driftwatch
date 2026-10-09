# DriftWatch: Detector Comparison

## Background

DriftWatch compares anomaly detectors for laptop telemetry under limited
ground truth. The goal is not to identify a universally best algorithm,
but to understand the different behaviors and trade-offs of each
detector.

The initial baselines were Robust Z-score and Mahalanobis Distance.
Additional candidates were evaluated to select complementary detectors
for sustained changes and multivariate anomalies.

## Candidate Comparison

### Multivariate detectors

Isolation Forest, One-Class SVM, and LOF were evaluated using engineered
features.

Isolation Forest achieved strong recall on some anomaly types but
produced excessive gaming false positives. LOF was retained as a
documented runner-up because of occasional recall improvements, but it
was less reliable on the gaming negative case.

One-Class SVM was selected after feature reduction and validation. A nu
sweep was performed, with the final configuration balancing anomaly
detection and gaming false positives.

### Sustained-change detectors

CUSUM and EWMA were compared on raw CPU usage.

CUSUM was selected because it detects accumulated deviations over time.
Tuning the slack parameter k was more important than increasing h: k
controlled whether normal noise accumulated into false alarms.

EWMA performed well on sustained drift but failed to recover short
events such as ram_spike because smoothing diluted short deviations.

## Final Detector Set

The final system contains:

-   Robust Z-score
-   Mahalanobis Distance
-   CUSUM
-   One-Class SVM

## Baseline Fitting Strategy

Baseline-fitting detectors exclude anomaly-labeled rows when estimating
normal behavior.

Known-normal heavy workloads, such as gaming_gpu_load, remain part of
the baseline because they represent legitimate system behavior.

## Production Notes

CUSUM is intentionally scoped to cpu_percent only. Applying the same
parameters to other metrics, especially GPU load, caused legitimate
gaming sessions to be flagged.

One-Class SVM and Mahalanobis use the validated production fit strategy
documented during evaluation.

## Results

Latest evaluation run (2026-10-09):

  Method          Real F1   Synthetic F1
  ------------- --------- --------------
  z-score           0.102          0.458
  Mahalanobis       0.431          0.523
  CUSUM             0.507          0.019
  OCSVM             0.426          0.301

## Findings

CUSUM performs poorly on synthetic data by design: it only observes CPU
changes, and its thresholds depend on the noise profile of the session.

Contaminated baselines reduce Mahalanobis performance. When injected
anomalies were included during baseline fitting, synthetic F1 dropped
from 0.523 to 0.208.

The battery_percent feature was removed after analysis showed it
represented session power state rather than anomaly behavior.

Gaming-session behavior on the real database: - Mahalanobis: 65/73
flagged - OCSVM: 18/73 flagged - CUSUM: 0/73 flagged

## Limitations

-   Ground truth is limited; unlabeled rows are treated as normal during
    evaluation.
-   Real false-positive rates may be overestimated because some normal
    labels may contain unknown abnormal behavior.
-   There are only a small number of confirmed real events.
-   Detector baselines are refit on each run, so historical
    classifications may change as additional data is collected.
-   CUSUM's single-metric scope limits detection of non-CPU anomalies.
