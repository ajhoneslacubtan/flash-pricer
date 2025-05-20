"""
Contain the application's configuration including the scenario configurations.

The configuration is run when starting the Orchestrator.
"""
import numpy as np
from algorithms.algorithms import (
    fit_feature_artifacts,
    load_orders,
    load_products,
    generate_flash_calendar,
    tag_flash_window,
    simulate_inventory,
    train_demand_model,
    build_price_grid,
    predict_units,
    compute_profit_surface,
    optimize_price,
    evaluate_flash_day,
    transform_features,
)
from taipy import Config, Scope

# -------------------------------------------------------------------
# 1. DataNodeConfigs
# -------------------------------------------------------------------

# Parameter: which seller’s catalog this scenario controls
seller_id_dn_cfg = Config.configure_data_node(
    id="seller_id_dn",
    storage_type="in_memory",
    scope=Scope.GLOBAL,
    default_data="6560211a19b47992c3666cc44a7e94c0"  # e.g. "6560211a19b47992c3666cc44a7e94c0"
)

def _no_write(*_, **__):  # read-only helper
    return None

def sample_lead_time():
    # e.g. uniform integer from 2 to 5 days
    return np.random.randint(2, 6)

orders_raw_dn_cfg = Config.configure_sql_data_node(
    id="orders_raw_dn",
    db_engine="sqlite",
    db_name="olist",              # just the filename
    sqlite_folder_path="data/",   # folder where it lives
    db_extra_args={},             # ensure no NoneType error
    read_query="""
        SELECT
            oi.order_id,
            o.order_purchase_timestamp AS ts,
            oi.product_id,
            oi.seller_id,
            oi.price,
            oi.freight_value
        FROM olist_order_items_dataset oi
        JOIN olist_orders_dataset o 
          ON oi.order_id = o.order_id
        WHERE o.order_status <> 'canceled';
    """,
    write_query_builder=_no_write,
    scope=Scope.GLOBAL,
)


products_raw_dn_cfg = Config.configure_sql_data_node(
    id="products_raw_dn",
    db_engine="sqlite",
    db_name="olist",
    sqlite_folder_path="data/",
    db_extra_args={},
    read_query="""
        SELECT
            p.product_id,
            t.product_category_name_english AS category_name,
            p.product_name_lenght,
            p.product_description_lenght,
            p.product_photos_qty,
            p.product_weight_g,
            p.product_length_cm,
            p.product_height_cm,
            p.product_width_cm
        FROM
        olist_products_dataset AS p
        LEFT JOIN
        product_category_name_translation AS t
            ON p.product_category_name = t.product_category_name;
    """,
    write_query_builder=_no_write,
    scope=Scope.GLOBAL,
)

# persisted outputs (parquet/pickle)
flash_calendar_dn_cfg      = Config.configure_data_node(id="flash_calendar_dn",          storage_type="parquet", scope=Scope.GLOBAL)
freight_median_by_cat_dn_cfg = Config.configure_data_node(id="freight_median_by_cat_dn", storage_type="parquet", scope=Scope.GLOBAL)
global_freight_median_dn_cfg = Config.configure_data_node(id="global_freight_median_dn", storage_type="pickle",  scope=Scope.GLOBAL)
top_categories_dn_cfg      = Config.configure_data_node(id="top_categories_dn",          storage_type="pickle",  scope=Scope.GLOBAL)
sku_encoder_dn_cfg         = Config.configure_data_node(id="sku_encoder_dn",             storage_type="pickle",  scope=Scope.GLOBAL)
features_dn_cfg            = Config.configure_data_node(id="features_dn",                storage_type="parquet", scope=Scope.GLOBAL)
model_dn_cfg               = Config.configure_data_node(id="model_dn",                   storage_type="pickle",  scope=Scope.GLOBAL)
feature_importance_dn_cfg  = Config.configure_data_node(id="feature_importance_dn",      storage_type="parquet", scope=Scope.GLOBAL)
metrics_dn_cfg             = Config.configure_data_node(id="metrics_dn",                 storage_type="pickle",  scope=Scope.GLOBAL)
rec_price_dn_cfg           = Config.configure_data_node(id="rec_price_dn",               storage_type="parquet", scope=Scope.GLOBAL)
kpi_dn_cfg                 = Config.configure_data_node(id="kpi_dn",                     storage_type="parquet", scope=Scope.GLOBAL)

# in-memory intermediates
price_grid_dn_cfg = Config.configure_data_node(
    id="price_grid_dn",     
    default_data={
        "min": 0.5,    # minimum price multiplier
        "max": 2.0,    # maximum price multiplier
        "n": 30        # number of price points to evaluate
    },
    storage_type="in_memory", 
    scope=Scope.GLOBAL
)
pred_units_dn_cfg      = Config.configure_data_node(id="pred_units_dn",             storage_type="in_memory", scope=Scope.GLOBAL)
profit_surface_dn_cfg  = Config.configure_data_node(id="profit_surface_dn",         storage_type="in_memory", scope=Scope.GLOBAL)
orders_df_dn_cfg        = Config.configure_data_node(id="orders_df_dn",             storage_type="in_memory", scope=Scope.GLOBAL)
products_df_dn_cfg      = Config.configure_data_node(id="products_df_dn",           storage_type="in_memory", scope=Scope.GLOBAL)
tagged_orders_df_dn_cfg = Config.configure_data_node(id="tagged_orders_df_dn",      storage_type="in_memory", scope=Scope.GLOBAL)
inventory_df_dn_cfg     = Config.configure_data_node(id="inventory_df_dn",          storage_type="in_memory", scope=Scope.GLOBAL)
price_grid_results_dn_cfg = Config.configure_data_node(id="price_grid_results_dn",  storage_type="in_memory", scope=Scope.GLOBAL)

# -------------------------------------------------------------------
# 3 ▸ Parameter DataNodes (in-memory with defaults)
# -------------------------------------------------------------------
flash_date_dn_cfg        = Config.configure_data_node("flash_date_dn",        default_data="2017-11-23")
synthetic_count_dn_cfg   = Config.configure_data_node("synthetic_count_dn",   default_data=5)
inv_method_dn_cfg        = Config.configure_data_node(
                                id="inv_method_dn",
                                default_data={
                                    # --- reorder policy ------------------------------------
                                    "method": "sQ",           # or "RS"
                                    "s"     : 20,             # reorder point
                                    "Q"     : 100,            # order quantity
                                    # "R": 7, "S": 300        # <- fields for an (R,S) policy
                                    # -------------------------------------------------------
                                    "initial_inventory": 50,
                                    # JSON description of the lead-time sampler
                                    "lead_time": {"type": "uniform_int", "low": 2, "high": 5}
                                },
                                scope=Scope.GLOBAL          # GLOBAL so every task can read it
                            )
unit_cost_factor_dn_cfg  = Config.configure_data_node("unit_cost_factor_dn",  default_data=0.60)
marketing_boost_dn_cfg   = Config.configure_data_node("marketing_boost_dn",   default_data=20)
model_type_dn_cfg        = Config.configure_data_node("model_type_dn",        default_data="xgboost")
objective_dn_cfg         = Config.configure_data_node("objective_dn",         default_data="profit")

# -------------------------------------------------------------------
# 2. TaskConfigs
# -------------------------------------------------------------------
load_orders_task_cfg = Config.configure_task(
    id="load_orders_task",
    function=load_orders,
    input=[orders_raw_dn_cfg, seller_id_dn_cfg],
    output=orders_df_dn_cfg,
    skippable=True,
)

load_products_task_cfg = Config.configure_task(
    id="load_products_task",
    function=load_products,
    input=[products_raw_dn_cfg, orders_df_dn_cfg],
    output=products_df_dn_cfg,
    skippable=True,
)

generate_flash_calendar_task_cfg = Config.configure_task(
    "generate_flash_calendar_task",
    generate_flash_calendar,
    input=[flash_date_dn_cfg, synthetic_count_dn_cfg],
    output=flash_calendar_dn_cfg,
    skippable=True,
)

tag_flash_window_task_cfg = Config.configure_task(
    id="tag_flash_window_task",
    function=tag_flash_window,
    input=[orders_df_dn_cfg, flash_calendar_dn_cfg],
    output=tagged_orders_df_dn_cfg,
    skippable=True,
)

simulate_inventory_task_cfg = Config.configure_task(
    "simulate_inventory_task",
    simulate_inventory,
    input=[tagged_orders_df_dn_cfg, flash_calendar_dn_cfg, inv_method_dn_cfg],
    output=inventory_df_dn_cfg,
    skippable=True,
)

fit_feature_artifacts_task_cfg = Config.configure_task(
    id="fit_feature_artifacts_task",
    function=fit_feature_artifacts,
    input=[inventory_df_dn_cfg, products_df_dn_cfg],
    output=[
        freight_median_by_cat_dn_cfg,
        global_freight_median_dn_cfg,
        top_categories_dn_cfg,
        sku_encoder_dn_cfg,
    ],
    skippable=True,
)

transform_features_task_cfg = Config.configure_task(
    id="transform_features_task",
    function=transform_features,
    input=[
        inventory_df_dn_cfg,
        products_df_dn_cfg,
        freight_median_by_cat_dn_cfg,
        global_freight_median_dn_cfg,
        top_categories_dn_cfg,
        sku_encoder_dn_cfg,
        marketing_boost_dn_cfg,
    ],
    output=[features_dn_cfg],
)


train_demand_model_task_cfg = Config.configure_task(
    "train_demand_model_task",
    train_demand_model,
    input=[features_dn_cfg, model_type_dn_cfg],
    output=[model_dn_cfg, feature_importance_dn_cfg, metrics_dn_cfg],
    skippable=True,
)

build_price_grid_task_cfg = Config.configure_task(
    "build_price_grid_task",
    build_price_grid,
    input=[
        features_dn_cfg,
        price_grid_dn_cfg,
    ],
    output=price_grid_results_dn_cfg,  # Changed from price_grid_dn_cfg
)

predict_units_task_cfg = Config.configure_task(
    id="predict_units_task",
    function=predict_units,
    input=[model_dn_cfg, price_grid_results_dn_cfg, features_dn_cfg],
    output=pred_units_dn_cfg,
)

compute_profit_surface_task_cfg = Config.configure_task(
    "compute_profit_surface_task",
    compute_profit_surface,
    input=[pred_units_dn_cfg, unit_cost_factor_dn_cfg, objective_dn_cfg],
    output=profit_surface_dn_cfg,
)

optimize_price_task_cfg = Config.configure_task(
    id="optimize_price_task",
    function=optimize_price,
    input=[profit_surface_dn_cfg, inventory_df_dn_cfg],
    output=rec_price_dn_cfg,
)

evaluate_flash_day_task_cfg = Config.configure_task(
    id="evaluate_flash_day_task",
    function=evaluate_flash_day,
    input=[rec_price_dn_cfg, features_dn_cfg],
    output=kpi_dn_cfg,
)

# -------------------------------------------------------------------
# ScenarioConfig (no parameters argument)
# -------------------------------------------------------------------
scenario_cfg = Config.configure_scenario(
    id="flash_pricing_scenario",
    task_configs=[
        load_orders_task_cfg,
        load_products_task_cfg,
        generate_flash_calendar_task_cfg,
        tag_flash_window_task_cfg,
        simulate_inventory_task_cfg,
        fit_feature_artifacts_task_cfg,
        transform_features_task_cfg,
        train_demand_model_task_cfg,
        build_price_grid_task_cfg,
        predict_units_task_cfg,
        compute_profit_surface_task_cfg,
        optimize_price_task_cfg,
        evaluate_flash_day_task_cfg,
    ],
)

# -------------------------------------------------------------------
# Sequences (prep, train, optimize, kpi)
# -------------------------------------------------------------------
scenario_cfg.add_sequences({
    "prep_seq": [
        load_orders_task_cfg,
        load_products_task_cfg,
        generate_flash_calendar_task_cfg,
        tag_flash_window_task_cfg,
        simulate_inventory_task_cfg,
        fit_feature_artifacts_task_cfg,
        transform_features_task_cfg,
    ],
    "train_seq": [train_demand_model_task_cfg],
    "optimize_seq": [
        build_price_grid_task_cfg,
        predict_units_task_cfg,
        # compute_profit_surface_task_cfg,
        # optimize_price_task_cfg,
    ],
    "kpi_seq": [evaluate_flash_day_task_cfg],
})


Config.export("configuration/config.toml")