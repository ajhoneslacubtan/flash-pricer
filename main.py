"""
This is the main entry point for the Flash Pricer application.
It initializes the Taipy Orchestrator and GUI, and sets up the pages for the application.
"""
# ─── Imports ─────────────────────────────────────────────────────────────────
import taipy as tp
from taipy import Orchestrator
from taipy.gui import Gui

# ─── Your Pages (imported from /pages/*.py + .md) ───────────────────────────
from pages import (
    root_page,
    dashboard,
    scenario_studio,
    run_history,
    model_insights,
    data_explorer,
)

# ─── Your Core Config & (optional) test-data seeder ─────────────────────────
from configuration.config import scenario_cfg
# from create_test_scenarios import create_test_scenarios

# 1. Light theme (from your light.scss vars)
light_theme = {
    "palette": {
        "mode": "light",
        # overall backgrounds
        "background": {
            "default": "#fffefe",  # $body-bg
            "paper":   "#FFFFFF"   # $content-bg
        },
        # text colors
        "text": {
            "primary":   "#4f4f4f",  # $body-color
            "secondary": "#4F4F4F"   # $secondary
        },
        # primary & accent
        "primary":   {"main": "#54B689"},  # $primary
        "secondary": {"main": "#4F4F4F"}   # $secondary
    }
}

# 2. Dark theme (from your dark.scss vars)
dark_theme = {
    "palette": {
        "mode": "dark",
        "background": {
            "default": "#121820",  # $body-bg
            "paper":   "#171e28"   # .card background override
        },
        "text": {
            "primary":   "#b8babd",  # $body-color
            "secondary": "#f3f3f4"   # $text-muted
        },
        "primary":   {"main": "#54B689"},  # same accent
        "secondary": {"main": "#4F4F4F"}   # same secondary
    }
}


# ─── Route map ──────────────────────────────────────────────────────────────
PAGES = {
    "/": root_page,
    "dashboard": dashboard,
    "scenario_studio": scenario_studio,
    "run_history": run_history,
    "model_insights": model_insights,
    "data_explorer": data_explorer,
}

if __name__ == "__main__":
    # start Taipy Core
    orchestrator = Orchestrator()
    orchestrator.run()

    # create & submit the default scenario
    default_scenario = tp.create_scenario(scenario_cfg)
    default_scenario.name = "Default Scenario"
    default_scenario.submit()

    # pre-read data nodes so your pages’ charts/tables don’t start empty
    flash_date     = default_scenario.flash_date_dn.read()
    price_grid     = default_scenario.price_grid_dn.read()
    unit_cost_factor = default_scenario.unit_cost_factor_dn.read()
    marketing_boost= default_scenario.marketing_boost_dn.read()
    inv_method     = default_scenario.inv_method_dn.read()
    synthetic_count = default_scenario.synthetic_count_dn.read()
    model_type     = default_scenario.model_type_dn.read()
    objective      = default_scenario.objective_dn.read()
    df_orders     = default_scenario.orders_df_dn.read()
    df_products   = default_scenario.products_df_dn.read()
    df_features   = default_scenario.features_dn.read()
    kpi_summary   = default_scenario.kpi_dn.read()
    profit_surface= default_scenario.profit_surface_dn.read()
    feature_importance = default_scenario.feature_importance_dn.read()
    model_metrics = default_scenario.metrics_dn.read()


    selected_scenario = default_scenario
    price_grid_values = [price_grid["min"], price_grid["max"]]
    price_grid_n = price_grid["n"]

    active_scenario = True  # flag to enable/disable scenario execution

    # launch the GUI with styling hooks
    gui = Gui(
        pages=PAGES
    )
    gui.run(
        title="Flash Pricer",
        light_theme=light_theme,
        dark_theme=dark_theme
    )
