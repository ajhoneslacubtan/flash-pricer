### Dynamic Flash-Pricing Optimizer — Project Specification

*(Taipy 4.0.3 end-to-end pipeline)*

---

#### 1  Business Goal

Build a **one-seller flash-campaign pricing tool** that  - forecasts demand,

* optimises price per SKU under inventory constraints, and
* quantifies profit/revenue lift.
  The pipeline must run on the public **Olist e-commerce dataset** (+ simulated inventory) and be explorable through a Taipy GUI.

---

#### 2  Technology Stack

| Layer         | Choice                                         | Notes                                                      |
| ------------- | ---------------------------------------------- | ---------------------------------------------------------- |
| Orchestration | **Taipy Core 4.0.3**                           | Scenario + Task + Sequence APIs                            |
| Data Science  | **Python 3.12**, pandas, NumPy, Numba, XGBoost | Linear fallback available                                  |
| Storage       | Parquet / Pickle                               | Persisted DataNodes in `data/output/`                      |
| UI            | **Taipy GUI** pages                            | Dashboard / Scenario Studio / Run History / Model Insights |
| Code Layout   | created with `taipy create`                    | See directory tree in first message                        |

---

#### 3  Dataset & Scope

*SQLite `olist.db`* tables filtered to **one seller** via `seller_id_dn`.
Date span: *2016-09 → 2018-09*.
All pipeline parameters are DataNodes (editable in GUI).

---

#### 4  Key Parameters (DataNodes)

| ID                     | Default                      | Purpose                                                  |
| ---------------------- | ---------------------------- | -------------------------------------------------------- |
| `flash_date_dn`        | `2017-11-24`                 | Anchor flash-campaign day (projected to every data-year) |
| `price_grid_params_dn` | `{min:0.50, max:1.20, n:30}` | Candidate price grid multipliers                         |
| `unit_cost_factor_dn`  | `0.60`                       | Cost = factor × price                                    |
| `inv_method_dn`        | Poisson *s,Q* policy         | Inventory simulation spec                                |
| `marketing_boost_dn`   | `100`                        | In-campaign marketing index                              |
| `model_type_dn`        | `"xgboost"`                  | `"linear"` optional                                      |
| `objective_dn`         | `"profit"`                   | Alternative `"revenue"`                                  |

---

#### 5  Execution Graph (Tasks)

```
LOAD  →  CALENDAR  →  TAG  →  SIM INVENTORY  →  FIT ARTIFACTS  →  FEATURES
                                                                      ↓
                    TRAIN MODEL  →  PRICE GRID  →  PREDICT UNITS  →  PROFIT SURFACE
                                                                                 ↓
                                               INVENTORY →  OPTIMISE PRICE  →  KPI EVAL
```

| Sequence             | Tasks                                                                                                                                                | Purpose                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **`prep_train_seq`** | load\_orders • load\_products • generate\_flash\_calendar • tag\_flash\_window • simulate\_inventory • fit\_feature\_artifacts • transform\_features | Compute features & artifacts for training          |
| **`train_seq`**      | train\_demand\_model                                                                                                                                 | Fit XGB / Linear; saves model & feature importance |
| **`optimize_seq`**   | build\_price\_grid • predict\_units • compute\_profit\_surface • optimize\_price                                                                     | Price recommendation workflow                      |
| **`kpi_seq`**        | evaluate\_flash\_day                                                                                                                                 | Actual vs Expected vs Baseline KPIs                |
| **`inference_seq`**  | All of *optimize\_seq* plus upstream loads but **skips** `fit_feature_artifacts`                                                                     | Fast scoring on a new flash date                   |

All compute-heavy tasks (`simulate_inventory`, `transform_features`, `train_demand_model`) are **skippable** if inputs unchanged.

---

#### 6  DataNode Catalogue (selected)

| ID                                  | Storage        | Scope  | Description                                              |
| ----------------------------------- | -------------- | ------ | -------------------------------------------------------- |
| `orders_raw_dn` / `products_raw_dn` | SQL            | GLOBAL | Raw tables (read-only)                                   |
| `features_dn`                       | Parquet        | GLOBAL | Final modelling table                                    |
| `model_dn`                          | Pickle         | GLOBAL | Trained regressor                                        |
| `profit_surface_dn`                 | In-mem         | GLOBAL | SKU × price lattice                                      |
| `rec_price_dn`                      | Parquet        | GLOBAL | Recommended price list                                   |
| `kpi_dn`                            | Parquet        | GLOBAL | KPI table for dashboard                                  |
| Artifact DNs                        | Parquet/Pickle | GLOBAL | `freight_median_by_cat`, `top_categories`, `sku_encoder` |

---

#### 7  Algorithm Highlights

| Function                  | Role                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------ |
| `generate_flash_calendar` | Projects *flash\_date* month/day into every data-year; adds holidays & synthetic windows.  |
| `simulate_inventory`      | Numba-accelerated *(s,Q)* or *(R,S)* stock simulation; writes `on_hand`.                   |
| `transform_features`      | Rolling price baselines, seasonality, lagged demand, category one-hot, SKU ordinal encode. |
| `train_demand_model`      | Temporal 80/20 split; returns model + importance + metrics.                                |
| `build_price_grid`        | Generates per-SKU candidate prices from dict params.                                       |
| `predict_units`           | Re-computes price-dependent features, respects `anchor_flag`.                              |
| `compute_profit_surface`  | Adds revenue/profit & `score` (profit or revenue).                                         |
| `optimize_price`          | Arg-max `score` under `predicted_units ≤ on_hand`.                                         |
| `evaluate_flash_day`      | KPI table: actual, baseline, expected; lift % and totals.                                  |

---

#### 8  GUI Pages (agreed)

1. **Dashboard** – headline KPIs, price lift heat-map.
2. **Scenario Studio** – parameter widgets; run sequences.
3. **Run History** – list & compare scenarios.
4. **Model Insights** – feature importance, training metrics (lowest priority).

---

#### 9  Usage Workflow

1. **Select seller** (`seller_id_dn`) → run *prep\_train\_seq* once.
2. Adjust parameters (price grid, cost factor, flash date).
3. Run *optimize\_seq* to get new recommendations instantly.
4. Inspect KPIs on Dashboard.
5. After real campaign, import actual sales → re-run *kpi\_seq* for truth.

---

#### 10  Next Sprint — UI Build

*Use this spec in a new chat session focused on Taipy GUI.*

* Bind each parameter DN to appropriate widgets (DatePicker, Slider, Dropdown).
* Trigger sequences with buttons; display Task status.
* Show `kpi_dn` totals in cards; per-SKU table in DataTable; profit-surface heat-map optional.
* Provide download buttons for `rec_price_dn` CSV.

---

**End of Specification**
