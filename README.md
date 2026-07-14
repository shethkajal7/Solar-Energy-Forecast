<div align="center">

<!-- Animated Solar Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,30&height=200&section=header&text=Solar%20PV%20Performance%20Analytics&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=AI-Powered%20Fault%20Detection%20%7C%20Maintenance%20Prediction%20%7C%20Generation%20Forecasting&descSize=15&descAlignY=58&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge)]()

<br/>

> **Turning raw inverter data into actionable solar plant intelligence.**  
> Built to mirror the analytical workflows used by industrial PV asset managers.

<br/>

```
🌞  38,376 daylight readings  ·  22 inverters  ·  34 days  ·  2 datasets merged
```

</div>

---

## 🔭 Project Overview

Solar power plants generate enormous amounts of operational data — but raw data alone doesn't prevent downtime or recover lost energy. This project applies data analytics and machine learning to **Plant 1**, an industrial solar facility in India, to extract three categories of operational intelligence:

1. **Which inverters are underperforming?** — and by how much, in quantified kWh terms
2. **When do panels need cleaning?** — detected from efficiency degradation patterns, not guesswork
3. **How much power will be generated tomorrow?** — to support grid balancing decisions

This mirrors the exact analytical workflows deployed by companies in their commercial Smart O&M platforms.

---

## 💼 Business Problems Solved

| # | Problem | Why It Matters |
|---|---------|----------------|
| 1 | Identify faulty or suboptimally performing equipment | Chronic underperformance loses thousands of kWh annually — invisible without data |
| 2 | Detect need for panel cleaning / maintenance | Dirty panels silently erode output; early detection prevents critical efficiency loss |
| 3 | Predict power generation for the next couple of days | Grid operators need generation forecasts for load balancing and energy trading |

---

## 📦 Dataset

**Source:** [Kaggle — Solar Power Generation Data (India)](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)

Two files for Plant 1 were used:

```
📁 data/
├── Plant_1_Generation_Data.csv       # Inverter-level power output
└── Plant_1_Weather_Sensor_Data.csv   # Plant-level sensor readings
```

### Generation Data
| Column | Description |
|--------|-------------|
| `DATE_TIME` | Timestamp (15-minute intervals) |
| `PLANT_ID` | Plant identifier (constant: 4135001) |
| `SOURCE_KEY` | Unique inverter ID (22 inverters) |
| `DC_POWER` | DC power output from solar panels (W) |
| `AC_POWER` | AC power after inverter conversion (W) |
| `DAILY_YIELD` | Cumulative daily energy yield (kWh) |
| `TOTAL_YIELD` | Lifetime energy yield (kWh) |

### Sensor Data
| Column | Description |
|--------|-------------|
| `DATE_TIME` | Timestamp (15-minute intervals) |
| `AMBIENT_TEMPERATURE` | Air temperature at plant (°C) |
| `MODULE_TEMPERATURE` | Solar panel surface temperature (°C) |
| `IRRADIATION` | Solar irradiation intensity (W/m²) |

### Data Quality Summary
```
✅ Zero missing values in both files
✅ No date gaps — complete 34-day coverage
✅ 3,157 common timestamps successfully merged
✅ 38,376 valid daylight rows after nighttime filtering (IRRADIATION > 0)
⚠️  1,047 negative electricity readings in raw data (explained: net grid export)
⚠️  ~47% zero power readings (explained: nighttime — physically correct)
```

---

## 🏗 Project Architecture

```
Raw Data
    │
    ▼
┌─────────────────────────────────────┐
│         DATA PREPARATION            │
│  • Parse & standardise datetimes    │
│  • Merge generation + sensor data   │
│  • Drop redundant columns           │
│  • Engineer time features           │
│  • Filter to daylight hours only    │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│                    THREE ANALYSES                         │
│                                                           │
│  ┌─────────────────┐  ┌──────────────────┐  ┌─────────┐  │
│  │ Fault Detection │  │ Maintenance Det. │  │Forecast │  │
│  │                 │  │                  │  │         │  │
│  │ Peer comparison │  │ Efficiency ratio │  │ LinReg  │  │
│  │ across 22       │  │ vs 7-day rolling │  │ Random  │  │
│  │ inverters       │  │ average baseline │  │ Forest  │  │
│  └─────────────────┘  └──────────────────┘  └─────────┘  │
└──────────────────────────────────────────────────────────┘
               │
               ▼
        Actionable Insights
```

---

## 🔍 Key Findings

### ⚡ Finding 1 — Faulty Inverter Detection

**Method:** Peer comparison — all inverters share identical weather conditions, so consistent underperformers indicate structural faults.

```python
# Performance ratio: how each inverter compares to plant average
inverter_avg['PERFORMANCE_RATIO'] = (inverter_avg['AVG_DC_POWER'] / plant_avg) * 100
```

**Results:**

| Inverter ID | Avg DC Power | Underperformance | Energy Lost (34 days) | Projected Annual Loss |
|-------------|-------------|-----------------|----------------------|----------------------|
| `bvBOhCH3iADSZry` | 5,091.55 W | **10.54%** | 262.79 kWh | **2,821 kWh/year** |
| `1BY6WEcLGh8j5v7` | 5,173.35 W | **9.10%** | 226.81 kWh | **2,435 kWh/year** |
| Healthy Average | 5,691.19 W | — | — | — |

> 💡 **Combined annual energy loss: ~5,256 kWh/year** from just 2 of 22 inverters.  
> Both inverters showed consistent underperformance across **all 34 days** — confirming structural fault, not weather-related variation.

**Possible root causes:**
- Degraded solar panel strings connected to these inverters
- Partial shading from nearby obstructions
- Loose wiring or connection degradation
- Inverter hardware degradation

---

### 🧹 Finding 2 — Panel Cleaning & Maintenance Detection

**Method:** Track the ratio of DC power output to irradiation daily. A progressive drop in this efficiency ratio (below a 7-day rolling baseline) signals panel soiling.

```python
daily_perf['efficiency_ratio'] = daily_perf['avg_dc_power'] / daily_perf['avg_irradiation']
daily_perf['rolling_efficiency'] = daily_perf['efficiency_ratio'].rolling(window=7, min_periods=1).mean()
```

**Correlation between irradiation and DC power: 0.97** ✅ — confirms irradiation is a valid baseline for expected output.

**Flagged Days:**

| Date | Efficiency Ratio | Drop vs Baseline | Status |
|------|-----------------|-----------------|--------|
| 2020-05-20 | 13,392.03 | 2.66% | ⚠️ Warning |
| 2020-05-21 | 13,086.75 | 4.21% | ⚠️ Warning |
| 2020-05-22 | 13,194.44 | 3.18% | ⚠️ Warning |
| 2020-05-24 | 13,051.62 | 2.99% | ⚠️ Warning |
| 2020-06-06 | 13,344.54 | 2.26% | ⚠️ Warning |
| 2020-06-07 | 12,808.82 | **5.13%** | 🔴 CRITICAL |

> 💡 The May 20–22 cluster and the June 6–7 consecutive drop are classic **progressive soiling signatures** in dry, dusty conditions. Early detection on May 20th could have prevented the June 7th critical breach.

---

### 📈 Finding 3 — Power Generation Forecasting

**Method:** Aggregate daily plant-level generation, train on 29 days, test on last 5 days.

**Features used:**
```
avg_irradiation     → solar energy input (dominant driver)
avg_module_temp     → panel surface temperature
avg_ambient_temp    → air temperature
day_number          → captures temporal trend
```

**Model Comparison:**

| Model | MAE | R² Score |
|-------|-----|---------|
| Linear Regression | 36,392 W | 0.7098 |
| **Random Forest** | **35,373 W** | **0.7541** ✅ |

**Feature Importance (Random Forest):**

```
avg_irradiation    ████████████████████████████  69.3%  ← dominant driver
avg_module_temp    ███████                        18.1%
day_number         ███                             8.5%
avg_ambient_temp   █                               4.1%
```

> 💡 **Irradiation drives 69.3% of generation.** This confirms that solar power forecasting is fundamentally a weather forecasting problem — accurate irradiation prediction unlocks accurate generation prediction.

---

## 🛠 Technical Approach

### Why Peer Comparison for Fault Detection?
Instead of using a fixed threshold (e.g., "flag if below 3,000W"), peer comparison dynamically adjusts for weather — on a cloudy day, all inverters produce less. This approach flags inverters that produce **less than their peers on the same day**, eliminating weather as a confounding factor.

### Why Rolling Average for Maintenance Detection?
A static efficiency baseline would miss seasonal drift. A 7-day rolling average creates a **dynamic local baseline** — if the plant gradually improves (e.g., after rain cleans panels naturally), the threshold moves with it. Only drops relative to the recent trend are flagged.

### Why Random Forest Over Linear Regression?
Linear regression assumes a linear relationship between features and output. Solar generation has non-linear effects (e.g., extreme heat reduces panel efficiency non-linearly). Random Forest captures these without overfitting on such a small dataset (34 days).

---

## ⚠️ Honest Limitations

Being transparent about limitations is part of good data science practice.

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| Only 34 days of data | No seasonality captured; forecasting model is limited | Collect 1–2 years for robust seasonal modelling |
| Single plant | Model may not generalise to other plants/geographies | Validate on Plant 2 before production deployment |
| No string-level data | Cannot pinpoint exact panel causing inverter fault | Integrate string-level monitoring |
| No external weather forecast | Forecasting uses historical weather, not future predictions | Integrate Solcast or NASA POWER API for true forecasting |
| India-specific dataset | Irradiance patterns differ by geography | Retrain model per region |

---
"I built a Random Forest forecasting model that achieved R² of 0.75 on the test window, but when I ran cross-validation the score dropped to 0.19 with high variance. This told me the model was getting lucky on those 5 days — 34 days simply isn't enough training data for a reliable forecast. In production I would need at minimum 1–2 years of data and replace same-day sensor readings with weather forecast API inputs like Solcast."

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
```

### Setup
```bash
# Clone the repository
git clone https://github.com/Manas-singh14/Solar-PV-power-generation-analysis.git
cd solar-pv-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
```

### Run the Analysis
```bash
# Launch Jupyter Notebook
jupyter notebook

# Or run the full pipeline as a script
python src/analysis.py
```

### Data Setup
```
Place the Kaggle dataset files in the data/ directory:
data/
├── Plant_1_Generation_Data.csv
└── Plant_1_Weather_Sensor_Data.csv
```

---

## 📁 Project Structure

```
solar-pv-analytics/
│
├── 📂 data/
│   ├── Plant_1_Generation_Data.csv
│   └── Plant_1_Weather_Sensor_Data.csv
│
├── 📂 notebooks/
│   ├── 01_data_preparation.ipynb        # Merging, cleaning, feature engineering
│   ├── 02_fault_detection.ipynb         # Inverter peer comparison analysis
│   ├── 03_maintenance_detection.ipynb   # Panel soiling detection
│   └── 04_forecasting.ipynb             # Power generation forecasting
│
├── 📂 src/
│   ├── data_prep.py                     # Data loading and preprocessing
│   ├── fault_detection.py               # Inverter analysis functions
│   ├── maintenance.py                   # Efficiency ratio analysis
│   ├── forecasting.py                   # ML model training and evaluation
│   └── visualisation.py                 # Reusable plotting functions
│
├── 📂 outputs/
│   ├── charts/                          # Generated visualisations
│   └── Solar_PV_Analysis_Report.docx    # Full project report
│
├── 📂 docs/
│   └── Solar_PV_Analysis_Report.docx
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🧠 What I Learned

**Domain Knowledge:**
- Understood the full hardware hierarchy: Solar Panels → Inverters → Grid
- Learned the difference between DC power (panel output) and AC power (inverter output) and what faults each reveals
- Understood why nighttime zeros and negative net electricity readings are physically valid, not data errors

**Analytical Skills:**
- Peer comparison is more robust than fixed thresholds for anomaly detection in weather-dependent systems
- Rolling baselines are more reliable than static ones for efficiency monitoring
- Feature importance analysis revealed that solar forecasting is 69% a weather problem

**Honest Data Science:**
- A 34-day dataset is insufficient for capturing seasonality — always disclose limitations
- Correlation (0.97) should be verified before using a variable as a baseline
- Data "sparsity" from nighttime zeros is not a flaw — understanding the domain prevents misdiagnosis

---

## 

<br/>


<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,30&height=100&section=footer" width="100%"/>

</div>
