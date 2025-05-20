"""
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
from algorithms.inventory_policy import InventoryMethod
from numba import njit, prange


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
def generate_flash_calendar(
    flash_date: str | datetime,
    synthetic_count: int = 5,
) -> pd.DataFrame:
    """
    Build a table of 24-hour flash windows for every year in the dataset.

    Returns
    -------
    pd.DataFrame
        Columns: ['start_ts', 'end_ts', 'label', 'origin']
    """
    # convert flash_date param to date object
    anchor_date = pd.to_datetime(flash_date).date()

    # study window (hard-coded 2016-2018 for Olist; extend if needed)
    years = [2016, 2017, 2018]

    starts: list[datetime] = []
    labels: list[str] = []
    origin: list[str] = []

    for y in years:
        # Real retail flash days
        for label, d in [
            ("black_friday", get_black_friday(y)),
            ("mothers_day", get_mothers_day(y)),
            ("fathers_day", get_fathers_day(y)),
        ]:
            starts.append(datetime.combine(d, datetime.min.time()))
            labels.append(label)
            origin.append("real")

        # Brazilian fixed retail peaks & holidays
        for m, d in SPECIAL_FIXED:
            date_obj = datetime(y, m, d).date()
            starts.append(datetime.combine(date_obj, datetime.min.time()))
            labels.append("fixed_peak")
            origin.append("real")

        # National holidays via holidays.BR
        for hol_date, _ in holidays.BR(years=[y]).items():
            starts.append(datetime.combine(hol_date, datetime.min.time()))
            labels.append("holiday")
            origin.append("real")

        # Synthetic extra windows
        rng = pd.date_range(
            start=datetime(y, 1, 1),
            end=datetime(y, 12, 31),
            freq="D",
        )
        synthetic_choices = random.sample(list(rng), k=synthetic_count)
        for d in synthetic_choices:
            starts.append(d.normalize())
            labels.append("synthetic")
            origin.append("synthetic")

    # user-supplied anchor flash date (if not already in list)
    if anchor_date not in [dt.date() for dt in starts]:
        starts.append(datetime.combine(anchor_date, datetime.min.time()))
        labels.append("anchor")
        origin.append("user")

    start_ts = pd.Series(starts, name="start_ts").sort_values().reset_index(drop=True)
    df = pd.DataFrame(
        {
            "start_ts": start_ts,
            "end_ts": start_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),
            "label": labels,
            "origin": origin,
        }
    )
    return df.drop_duplicates(subset=["start_ts"])


# ──────────────────────────────
# 3 ▸ Tag orders with flash_flag
# ──────────────────────────────
def tag_flash_window(
    orders_df: pd.DataFrame,
    flash_calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a boolean `flash_flag` column to each order row.

    Rules
    -----
    flash_flag = 1  if order timestamp falls between any
                    start_ts and end_ts in flash_calendar_df.

    Returns
    -------
    pd.DataFrame
        Same columns as orders_df plus 'flash_flag' (bool)
    """
    df = orders_df.copy()
    df["flash_flag"] = False

    # Build interval index for efficient lookup
    for _, row in flash_calendar_df.iterrows():
        mask = (df["ts"] >= row["start_ts"]) & (df["ts"] <= row["end_ts"])
        df.loc[mask, "flash_flag"] = True
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
                       inv_method_dict: dict) -> pd.DataFrame:

    # ------------------------------------------------------------------
    # 1 ▸ rebuild InventoryMethod from Taipy JSON-dict
    # ------------------------------------------------------------------
    policy = _dict_to_policy(inv_method_dict.copy())

    # ------------------------------------------------------------------
    # 2 ▸ pandas: daily aggregation & full calendar
    # ------------------------------------------------------------------
    df = tagged_orders_df_dn.copy()
    df['date'] = pd.to_datetime(df['ts']).dt.floor('D')
    daily = (
        df.groupby(['product_id', 'date'])
          .agg(
              demand        = ('date', 'size'),
              price         = ('price', 'mean'),
              freight_value = ('freight_value', 'mean'),
              flash_flag    = ('flash_flag', 'max')
          )
          .reset_index()
    )

    start, end  = daily['date'].min(), daily['date'].max()
    all_dates   = pd.date_range(start, end, freq='D')
    skus        = daily['product_id'].unique()

    # full grid → ensure zero-demand days are present
    full = (
        pd.MultiIndex.from_product([skus, all_dates],
                                   names=['product_id', 'date'])
          .to_frame(index=False)
          .merge(daily, on=['product_id', 'date'], how='left')
          .fillna({'demand': 0,
                   'flash_flag': False,
                   'price': np.nan,
                   'freight_value': np.nan})
    )

    # ------------------------------------------------------------------
    # 3 ▸ build per-SKU empirical demand pools & sample
    # ------------------------------------------------------------------
    n_days, n_sku = len(all_dates), len(skus)
    demand_mat = np.zeros((n_days, n_sku), dtype=np.int64)

    for idx, sku in enumerate(skus):
        hist_flash  = full[(full.product_id == sku) & (full.flash_flag)].demand.values
        hist_normal = full[(full.product_id == sku) & (~full.flash_flag)].demand.values
        if hist_flash.size == 0:   hist_flash  = hist_normal
        if hist_normal.size == 0:  hist_normal = np.array([0])

        sku_rows = full.product_id == sku
        for t, day in enumerate(all_dates):
            is_flash = full.loc[sku_rows & (full.date == day), 'flash_flag'].iat[0]
            demand_mat[t, idx] = np.random.choice(hist_flash if is_flash else hist_normal)

    # ------------------------------------------------------------------
    # 4 ▸ Numba core
    # ------------------------------------------------------------------
    if policy.method == "sQ":
        on_hand = _simulate_sQ(demand_mat,
                               s = policy.s, Q = policy.Q,
                               initial_inv = policy.initial_inventory,
                               lt_low = policy.lead_time_low,
                               lt_high= policy.lead_time_high)
    elif policy.method == "RS":
        on_hand = _simulate_RS(demand_mat,
                               R = policy.R, S = policy.S,
                               initial_inv = policy.initial_inventory,
                               lt_low = policy.lead_time_low,
                               lt_high= policy.lead_time_high)
    else:
        raise ValueError("Unknown policy method: " + policy.method)

    # ------------------------------------------------------------------
    # 5 ▸ tidy DataFrame back for ML / reporting
    # ------------------------------------------------------------------
    out = full[['product_id', 'date',
                'price', 'freight_value', 'flash_flag']].copy()
    # after merge, rows are ordered by product_id then date ⟶ we need the same order
    out = out.sort_values(['date', 'product_id']).reset_index(drop=True)
    out['on_hand'] = on_hand.ravel(order='F')    # Fortran order: date‐major
    out['demand']  = demand_mat.ravel(order='F')

    return out.reset_index(drop=True)

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
          .fillna(df['price'])
    )
    

    # Impute missing baseline prices by defaulting to the actual daily price
    df['baseline_price_30d'] = (df['baseline_price_30d'].fillna(df['price'])['price']
          .transform(lambda x: x.rolling(30, min_periods=1).median().shift(1))
    )
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
    df = df.drop(columns=['product_id','category_name'])
    df = df.sort_values(['sku_le','date']).reset_index(drop=True)

    return df


# ------------------------------------------------------------------
# 6 ▸ Model training (regression)
# ------------------------------------------------------------------
def train_demand_model(
    features_dn: pd.DataFrame,
    model_type: str = "xgboost",
    **_kwargs,
):
    """
    Fit a regression model predicting units_sold.
    Returns (model, feature_importance).
    """
    df = features_dn.copy()
    cutoff = df["date"].max() - pd.Timedelta(days=45)
    train_df = df[df["date"] <= cutoff]

    X = train_df[[
        "avg_price",
        "discount_pct",
        "baseline_price",
        "marketing_idx",
        "month",
        "day_of_week",
        "is_holiday",
    ]]
    y = train_df["units_sold"]

    if model_type == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        feat_imp = pd.Series(model.coef_, index=X.columns)
    else:
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=0,
        ).fit(X, y)
        feat_imp = pd.Series(model.feature_importances_, index=X.columns)

    return model, feat_imp.sort_values(ascending=False)


# ------------------------------------------------------------------
# 7 ▸ Price grid builder
# ------------------------------------------------------------------
def build_price_grid(
    features_dn: pd.DataFrame,
    price_grid_min: float,
    price_grid_max: float,
    price_grid_n: int,
    **_kwargs,
) -> pd.DataFrame:
    """
    Create SKU × candidate price grid for the flash-day snapshot.
    """
    flash_date = features_dn["date"].max()
    flash_df = (
        features_dn[features_dn["date"] == flash_date]
        .loc[:, ["product_id", "avg_price"]]
        .rename(columns={"avg_price": "baseline_price"})
    )

    price_points = np.linspace(price_grid_min, price_grid_max, price_grid_n)
    records = [
        {
            "product_id": row["product_id"],
            "candidate_price": round(row["baseline_price"] * mult, 2),
        }
        for _, row in flash_df.iterrows()
        for mult in price_points
    ]
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# 8 ▸ Demand prediction over grid
# ------------------------------------------------------------------
def predict_units(
    model_dn,
    price_grid_dn: pd.DataFrame,
    features_dn: pd.DataFrame,
    **_kwargs,
) -> pd.DataFrame:
    """
    Join features to the price grid and predict units_sold.
    """
    model = model_dn
    flash_date = features_dn["date"].max()
    base_feats = features_dn.loc[
        features_dn["date"] == flash_date,
        [
            "product_id",
            "baseline_price",
            "marketing_idx",
            "month",
            "day_of_week",
            "is_holiday",
            "inventory",
        ],
    ]

    grid = price_grid_dn.merge(base_feats, on="product_id", how="left")
    grid["discount_pct"] = (
        (grid["baseline_price"] - grid["candidate_price"])
        / grid["baseline_price"]
    )

    X = (
        grid.rename(columns={"candidate_price": "avg_price"})
            .loc[:, [
                "avg_price",
                "discount_pct",
                "baseline_price",
                "marketing_idx",
                "month",
                "day_of_week",
                "is_holiday",
            ]]
    )
    preds = model.predict(X)
    grid["pred_units"] = np.clip(preds, 0, None)
    return grid


# ------------------------------------------------------------------
# 9 ▸ Profit surface
# ------------------------------------------------------------------
def compute_profit_surface(
    pred_units_dn: pd.DataFrame,
    unit_cost_factor: float,
    objective: str = "profit",
    **_kwargs,
) -> pd.DataFrame:
    """
    Calculate profit or revenue for every SKU × price candidate.
    """
    df = pred_units_dn.copy()
    df["unit_cost"] = df["baseline_price"] * unit_cost_factor

    if objective == "revenue":
        df["metric"] = df["candidate_price"] * df["pred_units"]
    else:
        df["metric"] = (df["candidate_price"] - df["unit_cost"]) * df["pred_units"]

    return df


# ------------------------------------------------------------------
# 10 ▸ Price optimization
# ------------------------------------------------------------------
def optimize_price(
    profit_surface_dn: pd.DataFrame,
    inventory_df_dn: pd.DataFrame,
    **_kwargs,
) -> pd.DataFrame:
    """
    Arg-max metric per SKU, subject to pred_units ≤ inventory.
    Falls back to unconstrained max if none feasible.
    """
    surface = profit_surface_dn.copy()
    latest = inventory_df_dn["date"].max()
    inv = (
        inventory_df_dn[inventory_df_dn["date"] == latest]
        .loc[:, ["product_id", "inventory"]]
    )

    surface = surface.merge(inv, on="product_id", how="left")
    surface["inventory"].fillna(0, inplace=True)
    surface["feasible"] = surface["pred_units"] <= surface["inventory"]

    feasible = surface[surface["feasible"]]
    if feasible.empty:
        # fallback: take overall max per SKU
        idx = (
            surface.groupby("product_id")["metric"]
                   .idxmax()
                   .dropna()
                   .astype(int)
                   .tolist()
        )
        rec = surface.loc[idx]
    else:
        idx = (
            feasible.groupby("product_id")["metric"]
                    .idxmax()
                    .dropna()
                    .astype(int)
                    .tolist()
        )
        rec = feasible.loc[idx]

    rec = rec.loc[:, ["product_id", "candidate_price", "pred_units", "metric"]]
    return rec.rename(columns={
        "candidate_price": "recommended_price",
        "metric": "objective",
    })


# ------------------------------------------------------------------
# 11 ▸ KPI evaluation
# ------------------------------------------------------------------
def evaluate_flash_day(
    rec_price_dn: pd.DataFrame,
    features_dn: pd.DataFrame,
    unit_cost_factor: float,
    objective: str = "profit",
    **_kwargs,
) -> pd.DataFrame:
    """
    Compare recommended vs historical metric on flash date and report KPIs.
    """
    flash_date = features_dn["date"].max()
    hist = features_dn[features_dn["date"] == flash_date].merge(
        rec_price_dn,
        on="product_id",
        how="left",
        indicator=True,
    )

    # Historical metric
    hist["hist_metric"] = np.where(
        objective == "revenue",
        hist["avg_price"] * hist["units_sold"],
        (hist["avg_price"] - hist["baseline_price"] * unit_cost_factor)
        * hist["units_sold"],
    )
    # Recommended metric
    hist["rec_metric"] = np.where(
        objective == "revenue",
        hist["recommended_price"] * hist["pred_units"],
        (hist["recommended_price"] - hist["baseline_price"] * unit_cost_factor)
        * hist["pred_units"],
    )

    total_hist = hist["hist_metric"].sum()
    total_rec = hist["rec_metric"].sum()
    lift = total_rec - total_hist
    pct_lift = lift / total_hist if total_hist else np.nan
    coverage = hist["_merge"].eq("both").mean()  # fraction of SKUs recommended

    kpi = pd.DataFrame([{
        "flash_date": flash_date,
        "objective": objective,
        "historical_total": total_hist,
        "recommended_total": total_rec,
        "absolute_lift": lift,
        "percent_lift": pct_lift,
        "coverage": coverage,
    }])
    return kpi
