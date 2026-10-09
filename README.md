# DriftWatch

Continuous system telemetry logging and anomaly detection for laptop
hardware.

## Overview

DriftWatch collects Windows hardware telemetry and evaluates anomaly
detectors under limited ground truth availability.

The project focuses on comparing detector behavior rather than assuming
one algorithm is universally best.

## Requirements

-   Python environment with project dependencies installed
-   Windows system
-   LibreHardwareMonitor running for telemetry collection

## Data Collection

Start LibreHardwareMonitor, then run:

``` powershell
python -m src.collector
```

The collector records:

-   CPU, RAM, GPU utilization and temperatures
-   Disk and network activity
-   Battery status
-   Process information
-   Hardware context

Raw telemetry is stored in SQLite.

### Labels

An example `data/labels.csv` file is provided to demonstrate the expected format.
For new datasets, create your own labels based on the recorded events and update
the file accordingly.

## Pipeline

The complete analysis pipeline:

``` powershell
python pipeline.py --fresh
```

This runs:

1.  Feature engineering
2.  Label bridge construction
3.  Detector execution
4.  Evaluation
5.  Label overlap checks

Individual steps:

``` powershell
python -m src.features
python -m src.build_label_bridge
python -m src.detectors
python -m src.evaluate --save
```

Synthetic evaluation:

``` powershell
python -m src.features --db data\driftwatch_synthetic.db
python -m src.build_label_bridge --db data\driftwatch_synthetic.db
python -m src.detectors --db data\driftwatch_synthetic.db
python -m src.evaluate --db data\driftwatch_synthetic.db --save
```

## Detection Methods

The final detector set contains:

-   Robust Z-score
-   Mahalanobis Distance
-   CUSUM
-   One-Class SVM

Each detector represents a different strategy: - point-based deviation -
multivariate distance - sequential drift detection - novelty detection

## Project Structure

    src/
      collector.py
      database.py
      config.py
      features.py
      detectors.py
      evaluate.py

    data/
      driftwatch.db
      driftwatch_synthetic.db

    notebooks/
      bakeoff_synthetic.ipynb
      Natural_capture_validation.ipynb

## Dashboard

Run:

``` powershell
streamlit run src\dashboard.py
```

## Known Limitations

-   Real anomaly labels are limited and incomplete.
-   Excluding known anomalies from baseline fitting uses information
    that would not normally be available in a fully unsupervised
    deployment.
-   CUSUM currently observes cpu_percent only.
-   Detectors are refit on each run, so historical results may change as
    more data is collected.
-   Database write failures are logged but do not currently trigger the
    configured five-failure circuit breaker stop.
-   Hardware sensor availability depends on the system.
