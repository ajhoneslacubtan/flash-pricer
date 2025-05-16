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
import os
from datetime import timedelta

from algorithms import *

from taipy import Config
from taipy import Scope


# ###########################################################################
# PLACEHOLDER: Put your application's configurations here                   #
#                                                                           #
# Example:                                                                  #
# scenario_config = Config.configure_scenario("placeholder_scenario", [])   #
#                                                                           #
# Or create a config.toml file and load it like this:                       #
# def configure():                                                          #
#    Config.load("config/config.toml")                                      #
#    return Config.scenarios["scenario_configuration"]                      #
#                                                                           #
# Comment, remove or replace the previous lines with your own use case      #
# ###########################################################################

# === DataNode Configurations ===

# 1. Raw Orders DataNode (SQL) 🗄
#    Loads order and order-item details from the Olist SQLite database.
orders_raw_dn = Config.configure_sql_data_node(
    id="orders_raw_dn",
    db_name=os.path.join(os.path.dirname(__file__), "..", "data", "olist.db"),
    db_engine="sqlite",
    read_query="""
        SELECT o.order_id,
               o.order_purchase_timestamp,
               i.product_id,
               i.price,
               i.order_item_id,
               i.shipping_limit_date
        FROM orders o
        JOIN order_items i USING(order_id)
    """,
    scope=Scope.GLOBAL,
    db_extra_args={}  # ensure proper SQLite connection handling
)

# 2. Raw Products DataNode (SQL) 🗄
#    Loads product metadata for feature enrichment.
products_raw_dn = Config.configure_sql_data_node(
    id="products_raw_dn",
    db_name=os.path.join(os.path.dirname(__file__), "..", "data", "olist.db"),
    db_engine="sqlite",
    read_query="SELECT * FROM products",
    scope=Scope.GLOBAL,
    db_extra_args={}
)

# 3. Flash Calendar DataNode (In-Memory) ⏱
#    Holds the list of flash-sale windows (real + synthetic).
flash_calendar_dn = Config.configure_data_node(
    id="flash_calendar_dn",
    default_data=[],
    scope=Scope.GLOBAL,
    storage_type="in_memory"
)

# 4. Features DataNode (Scenario-Scoped, Parquet) 🔧
#    Aggregated product-day features for modeling.
features_dn = Config.configure_data_node(
    id="features_dn",
    scope=Scope.SCENARIO,
    storage_type="parquet",      # persistent across scenario runs until invalidated
    validity_period=timedelta(days=1)
)

# === Additional DataNode Configurations ===

# 5. Price Grid DataNode (Scenario-Scoped, In-Memory) 🎯
#    Candidate price grid for optimization task.
price_grid_dn = Config.configure_data_node(
    id="price_grid_dn",
    scope=Scope.SCENARIO,
    storage_type="in_memory"
)

# 6. Predicted Units DataNode (Scenario-Scoped, Parquet) 📈
#    Model predictions for each SKU-price combination.
pred_units_dn = Config.configure_data_node(
    id="pred_units_dn",
    scope=Scope.SCENARIO,
    storage_type="parquet"
)

# 7. Recommended Price DataNode (Scenario-Scoped, Parquet) 💡
#    Final recommended prices, expected units, and profit per SKU.
rec_price_dn = Config.configure_data_node(
    id="rec_price_dn",
    scope=Scope.SCENARIO,
    storage_type="parquet"
)

# 8. Profit Surface DataNode (Scenario-Scoped, In-Memory) 🔥
#    Full profit surface heatmap data for visualization.
profit_surface_dn = Config.configure_data_node(
    id="profit_surface_dn",
    scope=Scope.SCENARIO,
    storage_type="in_memory"
)

# 9. KPI DataNode (Scenario-Scoped, In-Memory) 🏆
#    Key metrics: actual vs. recommended revenue, profit, margin, and stock-out rate.
kpi_dn = Config.configure_data_node(
    id="kpi_dn",
    scope=Scope.SCENARIO,
    storage_type="in_memory"
)

# 10. Feature Importance DataNode (Scenario-Scoped, Parquet) 📊
#     Stores model feature importance for insights.
feature_importance_dn = Config.configure_data_node(
    id="feature_importance_dn",
    scope=Scope.SCENARIO,
    storage_type="parquet"
)