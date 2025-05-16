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
Contain the application's configuration including the scenario configurations.

The configuration is run when starting the Orchestrator.
"""

from taipy import Config, Scope

# 1. Raw orders with cancellation filter
Config.configure_data_node(
    id="orders_raw_dn",
    storage_type="sql",
    scope=Scope.GLOBAL,
    default_path="data/olist.db",
    read_query="""
        SELECT
            oi.order_id,
            o.order_purchase_timestamp AS ts,
            oi.product_id,
            oi.price,
            oi.freight_value
        FROM olist_order_items_dataset oi
        JOIN olist_orders_dataset o
          ON oi.order_id = o.order_id
        WHERE o.order_status <> 'canceled'
    """,
)

# 2. Product master attributes
Config.configure_data_node(
    id="products_raw_dn",
    storage_type="sql",
    scope=Scope.GLOBAL,
    default_path="data/olist.db",
    table_name="olist_products_dataset",
)

# 3. Flash calendar (persisted per scenario)
Config.configure_data_node(
    id="flash_calendar_dn",
    storage_type="parquet",
    scope=Scope.GLOBAL,
    default_path="data/output/flash_calendar_{scenario.id}.parquet",
)

# 4. Engineered feature table
Config.configure_data_node(
    id="features_dn",
    storage_type="parquet",
    scope=Scope.GLOBAL,
    default_path="data/output/features_{scenario.id}.parquet",
)

# 5. Trained demand model artifact
Config.configure_data_node(
    id="model_dn",
    storage_type="pickle",
    scope=Scope.GLOBAL,
    default_path="data/output/model_{scenario.id}.pkl",
)

# 6. Feature importance scores
Config.configure_data_node(
    id="feature_importance_dn",
    storage_type="parquet",
    scope=Scope.GLOBAL,
    default_path="data/output/feature_importance_{scenario.id}.parquet",
)

# 7. Candidate price grid (in-memory)
Config.configure_data_node(
    id="price_grid_dn",
    storage_type="in_memory",
    scope=Scope.GLOBAL,
)

# 8. Predicted units per price (in-memory)
Config.configure_data_node(
    id="pred_units_dn",
    storage_type="in_memory",
    scope=Scope.GLOBAL,
)

# 9. Recommended price & profit results
Config.configure_data_node(
    id="rec_price_dn",
    storage_type="parquet",
    scope=Scope.GLOBAL,
    default_path="data/output/rec_price_{scenario.id}.parquet",
)

# 10. Profit surface lattice (in-memory)
Config.configure_data_node(
    id="profit_surface_dn",
    storage_type="in_memory",
    scope=Scope.GLOBAL,
)

# 11. KPI snapshot (persisted per scenario)
Config.configure_data_node(
    id="kpi_dn",
    storage_type="parquet",
    scope=Scope.GLOBAL,
    default_path="data/output/kpi_{scenario.id}.parquet",
)