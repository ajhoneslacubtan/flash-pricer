"""
algorithms/algorithms.py

This file is designed to contain the various Python functions used to configure tasks.

The functions will be imported by the __init__.py file in this folder.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

import holidays
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from algorithms.inventory_policy import InventoryMethod
from numba import njit, prange

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ──────────────────────────────
# Helper date utilities
# ──────────────────────────────
def get_mothers_day(year: int):
    d = datetime(year, 5, 1)
    while d.weekday() != 6:  # first Sunday
        d += timedelta(days=1)
    d += timedelta(days=7)  # second Sunday of May
    return d.date()


def get_fathers_day(year: int):
    d = datetime(year, 8, 1)
    while d.weekday() != 6:  # first Sunday
        d += timedelta(days=1)
    d += timedelta(days=7)  # second Sunday of August
    return d.date()


def get_black_friday(year: int):
    d = datetime(year, 11, 30)
    while d.weekday() != 4:  # last Friday
        d -= timedelta(days=1)
    return d.date()


SPECIAL_FIXED = {
    (6, 12),   # Dia dos Namorados
    (10, 12),  # Nossa Senhora Aparecida
    (12, 24),  # Christmas Eve
    (12, 25),  # Christmas Day
}


# ------------------------------------------------------------------
# 1 ▸ Ingestion tasks
# ------------------------------------------------------------------
def load_orders(orders_raw_df: pd.DataFrame, seller_id: str) -> pd.DataFrame:
    """
    Ingest raw orders, parse timestamps, and (optionally) filter by seller.
    
    Parameters
    ----------
    orders_raw_df : pd.DataFrame
        DataFrame loaded via SQL DataNode; must include at least
        ['order_id', 'ts', 'product_id'] and ideally 'seller_id'.
    seller_id : str
        If non-empty, only keep orders for this seller.

    Returns
    -------
    pd.DataFrame
        Cleaned orders with parsed datetime and optional seller filter.
    """
    df = orders_raw_df.copy()
    # parse the timestamp column
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    # filter out by seller if requested and the column exists
    if seller_id and 'seller_id' in df.columns:
        df = df[df['seller_id'] == seller_id]
    return df


def load_products(
    products_raw_df: pd.DataFrame,
    orders_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Subset the full products table to only those products sold in the
    provided orders DataFrame.

    Parameters
    ----------
    products_raw_df : pd.DataFrame
        Full products master table.
    orders_df : pd.DataFrame
        The orders DataFrame already filtered (e.g., by seller).
        Must include 'product_id'.

    Returns
    -------
    pd.DataFrame
        Only the products that appear in orders_df.
    """
    # identify unique product IDs in the filtered orders
    product_ids = orders_df['product_id'].unique()
    # subset the products table
    filtered = products_raw_df[products_raw_df['product_id'].isin(product_ids)].copy()
    return filtered


# ──────────────────────────────
# 2 ▸ Flash-calendar generator
# ──────────────────────────────
SPECIAL_FIXED = {(6, 12), (10, 12), (12, 24), (12, 25)}

# ------------------------------------------------------------------------------
# Flash‑calendar generator with YEAR‑PROJECTION anchor logic
# ------------------------------------------------------------------------------

def generate_flash_calendar(
    flash_date: str | datetime,
    synthetic_count: int = 5,
) -> pd.DataFrame:
    """Return a DataFrame of 24‑h windows, projecting the user‑chosen
    *month/day* onto every data year so an anchor exists inside history.
    """
    anchor_full = pd.to_datetime(flash_date).date()  # e.g., 2025‑11‑28
    anchor_md   = (anchor_full.month, anchor_full.day)

    years = [2016, 2017, 2018]  # Olist span
    starts, labels, origins = [], [], []

    for y in years:
        # Real retail flash days
        for lbl, d in [
            ("black_friday", get_black_friday(y)),
            ("mothers_day",  get_mothers_day(y)),
            ("fathers_day",  get_fathers_day(y)),
        ]:
            starts.append(datetime.combine(d, datetime.min.time()))
            labels.append(lbl); origins.append("real")

        # Fixed peaks & national holidays
        for m, d in SPECIAL_FIXED:
            starts.append(datetime(y, m, d))
            labels.append("fixed_peak"); origins.append("real")
        for hol_date, _ in holidays.BR(years=[y]).items():
            starts.append(datetime.combine(hol_date, datetime.min.time()))
            labels.append("holiday"); origins.append("real")

        # Synthetic windows
        rng = pd.date_range(datetime(y,1,1), datetime(y,12,31), freq="D")
        for d in random.sample(list(rng), k=synthetic_count):
            starts.append(d.normalize()); labels.append("synthetic"); origins.append("synthetic")

        # Anchor projected into this year
        starts.append(datetime(y, *anchor_md))
        labels.append("anchor"); origins.append("user")

    df = pd.DataFrame({
        "start_ts": pd.Series(starts).sort_values().reset_index(drop=True)
    })
    df["label"]  = labels
    df["origin"] = origins
    df["end_ts"] = df["start_ts"] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    # keep one row per start_ts preferring 'anchor' > real > synthetic
    prio = {"anchor":0,"black_friday":1,"mothers_day":1,"fathers_day":1,
            "holiday":2,"fixed_peak":2,"synthetic":3}
    df["_rank"] = df["label"].map(prio)
    df = (df.sort_values(["start_ts","_rank"]).drop_duplicates("start_ts").drop(columns="_rank").reset_index(drop=True))
    return df

# ──────────────────────────────
# 3 ▸ Tag orders with flash_flag
# ──────────────────────────────
def tag_flash_window(orders_df: pd.DataFrame, flash_calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Add `flash_flag` (any window) and `anchor_flag` (user anchor) to each order."""
    df = orders_df.copy()
    df["flash_flag"]  = False
    df["anchor_flag"] = False

    for _, row in flash_calendar_df.iterrows():
        mask = (df["ts"] >= row["start_ts"]) & (df["ts"] <= row["end_ts"])
        df.loc[mask, "flash_flag"] = True
        if row["label"] == "anchor":
            df.loc[mask, "anchor_flag"] = True
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 4 ▸ Inventory simulation
# ------------------------------------------------------------------

def _dict_to_policy(d: dict) -> InventoryMethod:
    lt = d.pop("lead_time")
    if lt["type"] != "uniform_int":
        raise ValueError("Only 'uniform_int' lead-time supported in fast path.")
    return InventoryMethod(**d,
                           lead_time_low = lt["low"],
                           lead_time_high= lt["high"])

@njit(parallel=True, cache=True)
def _simulate_sQ(demand, s, Q, initial_inv, lt_low, lt_high):
    n_days, n_sku = demand.shape
    on_hand_out = np.empty_like(demand)

    # run each SKU independently ⇒ safe to parallelise
    for k in prange(n_sku):
        inv      = initial_inv
        pending  = 0                       # pipeline not yet received
        max_lt   = lt_high
        pipeline = np.zeros(n_days + max_lt + 1, dtype=np.int64)

        for t in range(n_days):
            # (1) receive
            inv     += pipeline[t]
            pending -= pipeline[t]

            # (2) ship demand
            d   = demand[t, k]
            sh  = d if inv >= d else inv
            inv -= sh
            on_hand_out[t, k] = inv

            # (3) reorder if needed
            if inv <= s:
                lt = np.random.randint(lt_low, lt_high + 1)
                arrive = t + lt
                pipeline[arrive] += Q
                pending += Q
    return on_hand_out

@njit(parallel=True, cache=True)
def _simulate_RS(demand, R, S, initial_inv, lt_low, lt_high):
    n_days, n_sku = demand.shape
    on_hand_out = np.empty_like(demand)

    for k in prange(n_sku):
        inv      = initial_inv
        pending  = 0
        max_lt   = lt_high
        pipeline = np.zeros(n_days + max_lt + 1, dtype=np.int64)

        for t in range(n_days):
            # (1) receive arrivals
            inv     += pipeline[t]
            pending -= pipeline[t]

            # (2) ship demand
            d   = demand[t, k]
            sh  = d if inv >= d else inv
            inv -= sh
            on_hand_out[t, k] = inv

            # (3) review every R days (t=0, R, 2R, …)
            if (t % R) == 0:
                need = S - (inv + pending)
                if need > 0:
                    lt = np.random.randint(lt_low, lt_high + 1)
                    arrive = t + lt
                    pipeline[arrive] += need
                    pending += need
    return on_hand_out


def simulate_inventory(tagged_orders_df_dn: pd.DataFrame,
                       flash_calendar_df: pd.DataFrame,
                       inv_method_dict: dict) -> pd.DataFrame:
    """Simulate on‑hand inventory and retain both flash_flag and anchor_flag.
    The anchor_flag isolates the user‑chosen campaign date for inference
    while flash_flag covers all special days for model training.
    """
    policy = _dict_to_policy(inv_method_dict.copy())

    # 1 ▸ daily aggregation including both flags
    df = tagged_orders_df_dn.copy()
    df['date'] = pd.to_datetime(df['ts']).dt.floor('D')
    daily = (
    df.groupby(['product_id', 'date'])
      .agg(
          demand        = ('date', 'size'),
          price         = ('price', 'mean'),
          freight_value = ('freight_value', 'mean'),
          flash_flag    = ('flash_flag',  'max'),
          anchor_flag   = ('anchor_flag', 'max')      # NEW
      )
      .reset_index()
    )


    # 2 ▸ ensure full calendar grid
    start, end = daily['date'].min(), daily['date'].max()
    all_dates  = pd.date_range(start, end, freq='D')
    skus       = daily['product_id'].unique()

    full = (
        pd.MultiIndex.from_product([skus, all_dates],
                                names=['product_id', 'date'])
        .to_frame(index=False)
        .merge(daily,
                on=['product_id', 'date'],
                how='left')
        .fillna({
            'demand': 0,
            'flash_flag': False,
            'anchor_flag': False,      # NEW
            'price': np.nan,
            'freight_value': np.nan
        })
    )

    anchor_windows = flash_calendar_df.query("label == 'anchor'")
    anchor_dates = pd.date_range(
        anchor_windows['start_ts'].min(), anchor_windows['end_ts'].max(), freq="D"
    ).date
    full["anchor_flag"] = full["date"].dt.date.isin(anchor_dates).astype(int)


    # 3 ▸ empirical demand pools
    n_days, n_sku = len(all_dates), len(skus)
    demand_mat = np.zeros((n_days, n_sku), dtype=np.int64)

    for idx, sku in enumerate(skus):
        hist_flash  = full[(full.product_id == sku) & (full.flash_flag)].demand.values
        hist_normal = full[(full.product_id == sku) & (~full.flash_flag)].demand.values
        if hist_flash.size == 0:
            hist_flash = hist_normal
        if hist_normal.size == 0:
            hist_normal = np.array([0])

        sku_rows = full.product_id == sku
        for t, day in enumerate(all_dates):
            is_flash = full.loc[sku_rows & (full.date == day), 'flash_flag'].iat[0]
            demand_mat[t, idx] = np.random.choice(hist_flash if is_flash else hist_normal)

    # 4 ▸ numba core (policy‑specific)
    if policy.method == 'sQ':
        on_hand = _simulate_sQ(
            demand_mat,
            s=policy.s, Q=policy.Q,
            initial_inv=policy.initial_inventory,
            lt_low=policy.lead_time_low,
            lt_high=policy.lead_time_high,
        )
    elif policy.method == 'RS':
        on_hand = _simulate_RS(
            demand_mat,
            R=policy.R, S=policy.S,
            initial_inv=policy.initial_inventory,
            lt_low=policy.lead_time_low,
            lt_high=policy.lead_time_high,
        )
    else:
        raise ValueError(f"Unknown policy method: {policy.method}")

    # 5 ▸ tidy back
    out = full[['product_id', 'date', 'price', 'freight_value', 'flash_flag', 'anchor_flag']].copy()
    out = out.sort_values(['date', 'product_id']).reset_index(drop=True)
    out['on_hand'] = on_hand.ravel(order='F')
    out['demand']  = demand_mat.ravel(order='F')
    return out

# ------------------------------------------------------------------
# 5 ▸ Feature engineering
# ------------------------------------------------------------------

def fit_feature_artifacts(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> tuple[pd.Series, float, list[str], LabelEncoder]:
    """
    Compute and persist FE artifacts:
      - freight median by category
      - global freight median
      - top-N categories
      - SKU LabelEncoder
    """
    # Merge to get category
    merged = inventory_df.merge(
        products_df[['product_id','category_name']],
        on='product_id', how='left'
    )
    # Freight medians by category
    freight_median_by_cat = (
        merged.groupby('category_name')['freight_value']
              .median()
    )
    # Global freight median
    global_freight_median = float(merged['freight_value'].median())

    # Top-N categories
    top_categories = (
        products_df['category_name']
                   .value_counts()
                   .nlargest(10)
                   .index
                   .tolist()
    )

    # SKU encoder
    encoder = LabelEncoder()
    encoder.fit(inventory_df['product_id'])

    return freight_median_by_cat, global_freight_median, top_categories, encoder


def transform_features(
    inventory_df: pd.DataFrame,
    products_df: pd.DataFrame,
    freight_median_by_cat: pd.Series,
    global_freight_median: float,
    top_categories: list[str],
    sku_encoder: LabelEncoder,
    marketing_boost: float,
) -> pd.DataFrame:
    """
    Apply FE using persisted artifacts:
    """
    df = inventory_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Merge product info
    df = df.merge(
        products_df[['product_id','category_name']],
        on='product_id', how='left'
    )

    # Freight imputation & missing indicator
    df['freight_missing'] = df['freight_value'].isna().astype(int)
    df['freight_value'] = df.apply(
        lambda row: freight_median_by_cat.get(row['category_name'], global_freight_median)
                    if pd.isna(row['freight_value']) else row['freight_value'],
        axis=1
    )

    # Price fill-forward/back
    df.sort_values(['product_id','date'], inplace=True)
    df['price'] = df.groupby('product_id')['price'].ffill().bfill()

    # Rolling baseline price (30-day median shifted by 1)
    df['baseline_price_30d'] = (
        df.groupby('product_id')['price']
          .transform(lambda x: x.rolling(30, min_periods=1).median().shift(1))
    )
    df["baseline_price_30d"] = df["baseline_price_30d"].fillna(df['price'])


    # Discount percentage
    df['discount_pct'] = 1 - df['price'] / df['baseline_price_30d']
    df['discount_pct'] = df['discount_pct'].fillna(0).clip(lower=0)

    # Inventory scarcity features
    df['inv_ratio'] = df['on_hand'] / (df['on_hand'] + df['demand']).replace({0: np.nan})
    df['inv_ratio'] = df['inv_ratio'].fillna(0)

    # Days to stock-out estimate (on_hand / 7-day avg demand)
    df['ma7_demand'] = (
        df.groupby('product_id')['demand']
          .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
    )
    df['days_to_oos_est'] = df['on_hand'] / df['ma7_demand'].replace({0: np.nan})
    df['days_to_oos_est'] = df['days_to_oos_est'].fillna(0)
    df.drop(columns=['ma7_demand'], inplace=True)

    # Seasonality encodings
    df['dow'] = df['date'].dt.weekday
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month'] = df['date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    df.drop(columns=['dow','month'], inplace=True)

    # Flash features
    df['flash_flag'] = df['flash_flag'].astype(int)
    df['discount_x_flash'] = df['discount_pct'] * df['flash_flag']

    # Marketing intensity
    df['marketing_idx'] = np.where(df['flash_flag'] == 1, marketing_boost, 20)

    # Lagged demand features
    df['lag1_units'] = df.groupby('product_id')['demand'].shift(1).fillna(0)
    df['lag7_units'] = (
        df.groupby('product_id')['demand']
          .transform(lambda x: x.rolling(7, min_periods=1).mean().shift(1))
          .fillna(0)
    )

    # Elasticity buckets one-hot
    elasticity_dummies = pd.get_dummies(
        df['discount_pct'].apply(
            lambda x: 'zero' if x == 0 else 'low' if x <= 0.05 else 'medium' if x <= 0.20 else 'high'
        ),
        prefix='elasticity'
    )
    df = pd.concat([df, elasticity_dummies], axis=1)

    # Category one-hot using top_categories list
    for cat in top_categories:
        df[f"cat_{cat}"] = (df['category_name'] == cat).astype(int)
    df['cat_other'] = (~df['category_name'].isin(top_categories)).astype(int)

    # SKU ordinal encoding
    df['sku_le'] = sku_encoder.transform(df['product_id'])

    # Final cleanup: drop identifiers
    df = df.drop(columns=['category_name'])
    df = df.sort_values(['sku_le','date']).reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# 6 ▸ Model training (regression)
# ------------------------------------------------------------------

def train_demand_model(
    features_df: pd.DataFrame,
    model_type: str = "xgboost"
) -> tuple[object, pd.DataFrame]:
    """
    Train a regression model to predict daily demand (units_sold).

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature table produced by transform_features_task.
        Must include 'date', 'demand', and all numeric predictors.
    model_type : str
        One of {'xgboost', 'linear'}.

    Returns
    -------
    model : fitted regression model
    importance_df : pd.DataFrame
        Columns: ['feature', 'importance'] sorted by descending importance.
    """
    # 1) Prepare X, y
    df = features_df.copy()
    df = df.sort_values("date")
    # drop non-feature columns
    X = df.drop(columns=["date", "product_id", "demand", "anchor_flag"])
    y = df["demand"].values

    # 2) Temporal train/validation split (e.g. last 20% as val)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 3) Fit the chosen model
    if model_type == "xgboost":
        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X_train, y_train)

    # 4) Validation diagnostics (optional logging)
    y_pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    # 5) Extract feature importances
    if model_type == "xgboost":
        importances = model.feature_importances_
    else:  # linear
        importances = np.abs(model.coef_)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    metrics = {
        "rmse": rmse,
        "train_size": len(X_train),
        "val_size":   len(X_val),
    }

    return model, importance_df, metrics


# ------------------------------------------------------------------
# 7 ▸ Price grid builder
# ------------------------------------------------------------------
def build_price_grid(
    features_df: pd.DataFrame,
    price_grid_params: dict
) -> pd.DataFrame:
    """
    Build a candidate price grid per SKU for the flash day,
    with parameters passed as a single dict.

    Parameters
    ----------
    features_df : pd.DataFrame
        The full feature table, must include at least:
        ['product_id', 'flash_flag', 'baseline_price_30d', 'price']
    price_grid_params : dict
        {
          "min": float,   # e.g. 0.5 for 50% of baseline
          "max": float,   # e.g. 1.2 for 120% of baseline
          "n": int        # number of grid points
        }

    Returns
    -------
    pd.DataFrame
        Columns: ['product_id', 'price']
        One row per (SKU, candidate price).
    """
    grid_min = price_grid_params["min"]
    grid_max = price_grid_params["max"]
    grid_n   = price_grid_params["n"]

    # Filter to flash-day rows to get baseline prices
    flash_df = features_df[features_df["flash_flag"] == 1]
    base_prices = (
        flash_df[["product_id", "baseline_price_30d"]]
        .drop_duplicates(subset="product_id")
        .set_index("product_id")["baseline_price_30d"]
    )

    rows = []
    for sku, base in base_prices.items():
        # fallback if baseline is missing or nonpositive
        if pd.isna(base) or base <= 0:
            base = flash_df.loc[flash_df["product_id"] == sku, "price"].median()

        candidates = np.linspace(
            grid_min * base,
            grid_max * base,
            num=grid_n
        )
        df_grid = pd.DataFrame({
            "product_id": sku,
            "price": candidates
        })
        rows.append(df_grid)

    price_grid_df = pd.concat(rows, ignore_index=True)
    return price_grid_df


# ------------------------------------------------------------------
# 8 ▸ Demand prediction over grid
# ------------------------------------------------------------------
def predict_units(
    model,
    price_grid_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Predict units sold for each candidate price on the *anchor* flash day.

    Parameters
    ----------
    model : object
        Trained regression model with a `.predict(X)` method.
    price_grid_df : pd.DataFrame
        Candidate prices. Columns: ['product_id', 'price'].
    features_df : pd.DataFrame
        Feature table that *includes* the boolean columns `flash_flag` and
        `anchor_flag`.  Only rows where `anchor_flag == 1` are used for
        inference.

    Returns
    -------
    pd.DataFrame
        Columns ['product_id', 'price', 'predicted_units'] for the anchor day.
    """
    # ------------------------------------------------------------------
    # 1 ▸ Filter to anchor‑day snapshot (one row per SKU)
    # ------------------------------------------------------------------
    anchor_df = (
        features_df[features_df["anchor_flag"] == 1]
        .sort_values(["product_id", "date"])
        .drop_duplicates(subset=["product_id"], keep="last")
    )

    if anchor_df.empty:
        raise ValueError("No rows with anchor_flag == 1 found. Check flash_date and calendar tagging.")

    # ------------------------------------------------------------------
    # 2 ▸ Merge grid with anchor features
    # ------------------------------------------------------------------
    df = price_grid_df.merge(
        anchor_df,
        on="product_id",
        how="left",
        suffixes=("_candidate", "_orig"),
    )

    # ------------------------------------------------------------------
    # 3 ▸ Override price with candidate price & recompute price‑dependent feats
    # ------------------------------------------------------------------
    df["price"] = df["price_candidate"]
    df.drop(columns=[col for col in ["price_candidate", "price_orig"] if col in df.columns], inplace=True)

    # Recompute discount percentage and its flash interaction
    df["discount_pct"] = 1 - df["price"] / df["baseline_price_30d"].replace({0: np.nan})
    df["discount_pct"].fillna(0, inplace=True)
    df["discount_pct"] = df["discount_pct"].clip(lower=0)
    df["discount_x_flash"] = df["discount_pct"] * df["flash_flag"]  # flash_flag==1 for anchor rows

    # ------------------------------------------------------------------
    # 4 ▸ Assemble model input matrix X
    # ------------------------------------------------------------------
    drop_cols = [
        "date", "demand", "anchor_flag", "product_id",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --- ensure column order matches training ---
    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]          # reorder or raise KeyError
    # -------------------------------------------

    # 5 ▸ Predict
    df["predicted_units"] = model.predict(X)

    # ------------------------------------------------------------------
    # 5 ▸ Predict units
    # ------------------------------------------------------------------
    df["predicted_units"] = model.predict(X)

    return df[["product_id", "price", "predicted_units"]]


# ------------------------------------------------------------------
# 9 ▸ Profit surface
# ------------------------------------------------------------------
def compute_profit_surface(
    pred_units_df: pd.DataFrame,
    unit_cost_factor: float,
    objective: str = "profit",
) -> pd.DataFrame:
    """Convert predicted units to revenue/profit and objective score.

    Parameters
    ----------
    pred_units_df : pd.DataFrame
        Must contain ['product_id', 'price', 'predicted_units'].
    unit_cost_factor : float
        Cost as fraction of price (0 ≤ factor < 1).
    objective : str, default 'profit'
        'profit' or 'revenue' – column to optimise.

    Returns
    -------
    pd.DataFrame with columns
        ['product_id', 'price', 'predicted_units', 'revenue', 'profit', 'score']
    """
    if not 0 <= unit_cost_factor < 1:
        raise ValueError("unit_cost_factor must be in [0, 1). Got " f"{unit_cost_factor}")

    df = pred_units_df.copy()
    # sanitize demand
    df["predicted_units"] = df["predicted_units"].fillna(0).clip(lower=0)

    df["revenue"] = df["price"] * df["predicted_units"]
    contribution_margin = 1.0 - unit_cost_factor
    df["profit"] = df["revenue"] * contribution_margin

    if objective == "profit":
        df["score"] = df["profit"]
    elif objective == "revenue":
        df["score"] = df["revenue"]
    else:
        raise ValueError("objective must be 'profit' or 'revenue'.")

    # sort for convenience
    df = df.sort_values(["product_id", "score"], ascending=[True, False]).reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 10 ▸ Price optimization
# ------------------------------------------------------------------
def optimize_price(
    profit_surface_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """Select arg‑max score price for each SKU subject to stock.

    Parameters
    ----------
    profit_surface_df : DataFrame
        From compute_profit_surface – one row per (SKU, price).
    inventory_df : DataFrame
        Must include anchor‑day rows with ['product_id','on_hand'].

    Returns
    -------
    DataFrame
        Columns ['product_id','recommended_price','predicted_units',
                 'on_hand','expected_revenue','expected_profit','score']
    """
    # Keep only anchor-day inventory rows (anchor_flag==1)
    inv_anchor = inventory_df[inventory_df.get("anchor_flag", 1) == 1]
    stock = inv_anchor.set_index("product_id")["on_hand"]

    # Merge stock into profit surface
    df = profit_surface_df.merge(stock, on="product_id", how="left", suffixes=("", "_stock"))
    df["on_hand"].fillna(np.inf, inplace=True)  # if no info assume unlimited stock

    # Feasible rows: units ≤ stock
    feasible = df[df["predicted_units"] <= df["on_hand"]]

    # For SKUs where no price meets stock, keep row with smallest demand
    fallback = (
        df.loc[df.groupby("product_id")["predicted_units"].idxmin()]
          .assign(feasible=False)
    )
    feasible["feasible"] = True
    union = pd.concat([feasible, fallback[~fallback.index.isin(feasible.index)]], ignore_index=True)

    # Pick arg‑max score, tie‑break by higher price
    union.sort_values(["product_id", "feasible", "score", "price"],
                      ascending=[True, False, False, False], inplace=True)
    best = union.drop_duplicates("product_id", keep="first")

    out = best.rename(columns={
        "price": "recommended_price",
        "revenue": "expected_revenue",
        "profit": "expected_profit",
    })[
        [
            "product_id",
            "recommended_price",
            "predicted_units",
            "on_hand",
            "expected_revenue",
            "expected_profit",
            "score",
        ]
    ].reset_index(drop=True)

    return out


# ------------------------------------------------------------------------------
# evaluate_flash_day  ─ Actual  vs  Baseline  vs  Expected
# ------------------------------------------------------------------------------

def evaluate_flash_day(
    rec_price_df: pd.DataFrame,
    features_df: pd.DataFrame,
    unit_cost_factor: float,
) -> pd.DataFrame:
    """
    Build KPI table with three reference points for the anchor flash date
    (one row per SKU + TOTAL):

        • Actual   – what really happened historically (may be zero)
        • Baseline – model’s forecast at baseline_price_30d
        • Expected – optimiser’s recommendation (already in rec_price_df)

    Baseline KPIs do *not* need a second model run: they reuse the
    predicted_units from rec_price_df and scale them by the price ratio.
    """

    if not 0 <= unit_cost_factor < 1:
        raise ValueError("unit_cost_factor must be in [0,1).")

    margin = 1.0 - unit_cost_factor

    # ── 1 ▸ Anchor-day snapshot from features_df ─────────────────────────
    anchor = (
        features_df[features_df["anchor_flag"] == 1]
        .sort_values(["product_id", "date"])
        .drop_duplicates("product_id", keep="last")
        .reset_index(drop=True)
    )

    # Keep only columns we need
    anchor = anchor[[
        "product_id",
        "price",
        "demand",
        "baseline_price_30d"
    ]].rename(columns={
        "price":   "actual_price",
        "demand":  "actual_units",
        "baseline_price_30d": "baseline_price"
    })

    # If baseline_price is nan/<=0, fall back to actual_price
    anchor["baseline_price"].fillna(anchor["actual_price"], inplace=True)
    anchor.loc[anchor["baseline_price"] <= 0, "baseline_price"] = anchor["actual_price"]

    # ── 2 ▸ Merge optimiser output ──────────────────────────────────────
    kpi = rec_price_df.merge(anchor, on="product_id", how="left")

    # Replace NaNs for SKUs with zero historical sales
    kpi[["actual_units"]]     = kpi[["actual_units"]].fillna(0)
    kpi[["actual_price"]]     = kpi[["actual_price"]].fillna(kpi["recommended_price"])
    kpi[["baseline_price"]]   = kpi[["baseline_price"]].fillna(kpi["recommended_price"])

    # ── 3 ▸ Compute KPIs for each reference point ───────────────────────
    kpi["actual_revenue"]   = kpi["actual_price"]   * kpi["actual_units"]
    kpi["actual_profit"]    = kpi["actual_revenue"] * margin

    # **Baseline**: scale predicted_units by price elasticity proxy
    # Simple proxy: assume demand is roughly inversely proportional to price
    price_ratio            = kpi["baseline_price"] / kpi["recommended_price"]
    kpi["baseline_units"]  = kpi["predicted_units"] * price_ratio.clip(lower=0.2, upper=2.0)
    kpi["baseline_revenue"] = kpi["baseline_price"] * kpi["baseline_units"]
    kpi["baseline_profit"]  = kpi["baseline_revenue"] * margin

    # ── 4 ▸ Lift metrics ────────────────────────────────────────────────
    kpi["lift_vs_actual_rev_pct"]  = np.where(
        kpi["actual_revenue"] > 0,
        (kpi["expected_revenue"] - kpi["actual_revenue"]) / kpi["actual_revenue"],
        np.nan
    )
    kpi["lift_vs_baseline_rev_pct"] = np.where(
        kpi["baseline_revenue"] > 0,
        (kpi["expected_revenue"] - kpi["baseline_revenue"]) / kpi["baseline_revenue"],
        np.nan
    )
    kpi["lift_vs_actual_prof_pct"] = np.where(
        kpi["actual_profit"] > 0,
        (kpi["expected_profit"] - kpi["actual_profit"]) / kpi["actual_profit"],
        np.nan
    )
    kpi["lift_vs_baseline_prof_pct"] = np.where(
        kpi["baseline_profit"] > 0,
        (kpi["expected_profit"] - kpi["baseline_profit"]) / kpi["baseline_profit"],
        np.nan
    )

    # ── 5 ▸ TOTAL aggregate row ─────────────────────────────────────────
    totals = {
        "product_id": "_TOTAL_",
        "recommended_price": np.nan,
        "predicted_units":      kpi["predicted_units"].sum(),
        "on_hand":              kpi["on_hand"].sum(),
        "expected_revenue":     kpi["expected_revenue"].sum(),
        "expected_profit":      kpi["expected_profit"].sum(),
        "actual_revenue":       kpi["actual_revenue"].sum(),
        "actual_profit":        kpi["actual_profit"].sum(),
        "baseline_revenue":     kpi["baseline_revenue"].sum(),
        "baseline_profit":      kpi["baseline_profit"].sum(),
    }
    totals["lift_vs_actual_rev_pct"]   = (
        (totals["expected_revenue"] - totals["actual_revenue"]) /
        totals["actual_revenue"] if totals["actual_revenue"] else np.nan
    )
    totals["lift_vs_baseline_rev_pct"] = (
        (totals["expected_revenue"] - totals["baseline_revenue"]) /
        totals["baseline_revenue"] if totals["baseline_revenue"] else np.nan
    )
    totals["lift_vs_actual_prof_pct"]  = (
        (totals["expected_profit"] - totals["actual_profit"]) /
        totals["actual_profit"] if totals["actual_profit"] else np.nan
    )
    totals["lift_vs_baseline_prof_pct"] = (
        (totals["expected_profit"] - totals["baseline_profit"]) /
        totals["baseline_profit"] if totals["baseline_profit"] else np.nan
    )

    kpi_df = pd.concat([kpi, pd.DataFrame([totals])], ignore_index=True)
    return kpi_df
