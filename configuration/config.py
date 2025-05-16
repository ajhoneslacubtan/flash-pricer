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
# configuration/config.py
from pathlib import Path

from taipy import Config
from taipy.core.config import Scope  # explicit import keeps linters happy


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

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"

orders_raw_dn_cfg = Config.configure_sql_data_node(
    id="orders_raw_dn",
    db_engine="sqlite",
    db_name=str(DB_PATH),
    read_query="SELECT * FROM olist_order_items_dataset",  # adjust to real table
    scope=Scope.GLOBAL,
    cacheable=True,         # expensive to read; cache once
)

products_raw_dn_cfg = Config.configure_sql_data_node(
    id="products_raw_dn",
    db_engine="sqlite",
    db_name=str(DB_PATH),
    read_query="SELECT * FROM olist_products_dataset",
    scope=Scope.GLOBAL,
    cacheable=True,
)

flash_calendar_dn_cfg = Config.configure_data_node(
    id="flash_calendar_dn",
    storage_type="pickle",   # small Python list/dict → pickle is fine
    default_data=None,       # will be filled by a Task at first run
    scope=Scope.GLOBAL,
)

# ----------------------------------------------------------------------
# 2.  ENGINEERED FEATURES & MODEL  (re-usable across scenarios)
# ----------------------------------------------------------------------
features_dn_cfg = Config.configure_data_node(
    id="features_dn",
    storage_type="pickle",   # large DataFrame; pickle faster than csv
    cacheable=True,          # recompute only when upstream changes
    validity_period="P7D",   # invalidate after 7 days if you rerun project
    scope=Scope.GLOBAL,
)

model_dn_cfg = Config.configure_data_node(
    id="model_dn",
    storage_type="pickle",   # sklearn / xgboost model artifact
    scope=Scope.GLOBAL,
)

feature_importance_dn_cfg = Config.configure_data_node(
    id="feature_importance_dn",
    storage_type="json",
    scope=Scope.GLOBAL,
)

# ----------------------------------------------------------------------
# 3.  SCENARIO-SPECIFIC ARTIFACTS  (change with each what-if run)
# ----------------------------------------------------------------------
price_grid_dn_cfg = Config.configure_data_node(
    id="price_grid_dn",
    storage_type="in_memory",
    scope=Scope.SCENARIO,
)

pred_units_dn_cfg = Config.configure_data_node(
    id="pred_units_dn",
    storage_type="in_memory",
    scope=Scope.SCENARIO,
)

profit_surface_dn_cfg = Config.configure_data_node(
    id="profit_surface_dn",
    storage_type="in_memory",
    scope=Scope.SCENARIO,
)

rec_price_dn_cfg = Config.configure_data_node(
    id="rec_price_dn",
    storage_type="pickle",   # keep for audit / comparison
    scope=Scope.SCENARIO,
)

kpi_dn_cfg = Config.configure_data_node(
    id="kpi_dn",
    storage_type="json",
    scope=Scope.SCENARIO,
)