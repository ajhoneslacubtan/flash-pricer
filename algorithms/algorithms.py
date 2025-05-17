# Copyright 2021-2024 Avaiga Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""
This file is designed to contain the various Python functions used to configure tasks.

The functions will be imported by the __init__.py file in this folder.
"""
from __future__ import annotations

import datetime as dt
import importlib
from typing import List, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import random
import holidays

from xgboost import XGBRegressor




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
    return df


# ------------------------------------------------------------------
# 4 ▸ Inventory simulation
# ------------------------------------------------------------------
def simulate_inventory(
    tagged_orders_df_dn: pd.DataFrame,
    inv_method: str = "poisson",
    **_kwargs,
) -> pd.DataFrame:
    """
    Produce a per-SKU, per-day inventory table.
    Modes:
      • poisson – start_stock ~ Poisson(lambda = mean_daily_sales * 1.5)
      • custom_function – calls custom_inventory.simulate_inventory(df)
    """
    df = tagged_orders_df_dn.copy()
    df["date"] = df["ts"].dt.floor("D")

    if inv_method == "custom_function":
        try:
            custom_mod = importlib.import_module("custom_inventory")
            return custom_mod.simulate_inventory(df)
        except ModuleNotFoundError as exc:
            raise RuntimeError("custom_inventory module not found") from exc

    # Basic Poisson simulation:
    daily_sales = (
        df.groupby(["product_id", "date"])
          .size()
          .rename("units_sold")
          .reset_index()
    )
    mean_sales = (
        daily_sales.groupby("product_id")["units_sold"]
                   .mean()
                   .rename("mu")
                   .reset_index()
    )
    rng = np.random.default_rng(123)
    mean_sales["start_stock"] = mean_sales["mu"].apply(
        lambda mu: max(300, rng.poisson(mu * 1.5))
    )

    inv_records: List[dict] = []
    for _, row in mean_sales.iterrows():
        sku = row["product_id"]
        stock = int(row["start_stock"])
        sku_sales = daily_sales[
            daily_sales["product_id"] == sku
        ].sort_values("date")
        for _, sale_row in sku_sales.iterrows():
            inv_records.append({
                "product_id": sku,
                "date": sale_row["date"],
                "inventory": stock,
            })
            stock = max(stock - int(sale_row["units_sold"]), 0)

    inv_df = pd.DataFrame(inv_records)
    return inv_df


# ------------------------------------------------------------------
# 5 ▸ Feature engineering
# ------------------------------------------------------------------
def engineer_features(
    tagged_orders_df_dn: pd.DataFrame,
    products_df_dn: pd.DataFrame,
    inventory_df_dn: pd.DataFrame,
    marketing_boost: float = 20,
    **_kwargs,
) -> pd.DataFrame:
    """
    Aggregate to product-date level with explanatory features.
    Missing inventory ⇒ assumed infinite (99_999).
    """
    orders = tagged_orders_df_dn.copy()
    orders["date"] = orders["ts"].dt.floor("D")

    # Base aggregations
    agg = (
        orders.groupby(["product_id", "date"])
              .agg(
                  units_sold=("order_id", "count"),
                  avg_price=("price", "mean"),
                  flash_flag=("flash_flag", "max"),
              )
              .reset_index()
    )

    # Baseline (non-flash) median price per SKU
    baseline_price = (
        agg.query("flash_flag == 0")
           .groupby("product_id")["avg_price"]
           .median()
           .rename("baseline_price")
           .reset_index()
    )
    agg = agg.merge(baseline_price, on="product_id", how="left")
    agg["baseline_price"].fillna(agg["avg_price"], inplace=True)

    # Discount percentage
    agg["discount_pct"] = (
        (agg["baseline_price"] - agg["avg_price"])
        / agg["baseline_price"].replace(0, np.nan)
    ).fillna(0)

    # Join inventory
    features = agg.merge(
        inventory_df_dn,
        on=["product_id", "date"],
        how="left",
    )

    # Seasonality dummies
    features["month"]       = features["date"].dt.month
    features["day_of_week"] = features["date"].dt.dayofweek
    # features["is_holiday"]  = features["date"].isin(BR_HOLIDAYS).astype(int)

    # Marketing index
    features["marketing_idx"] = np.where(
        features["flash_flag"] == 1,
        marketing_boost,
        marketing_boost / 4,
    )

    features["inventory"] = features["inventory"].fillna(99_999)

    return features


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
