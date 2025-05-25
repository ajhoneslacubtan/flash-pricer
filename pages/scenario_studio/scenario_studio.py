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
A page of the application.
Page content is imported from the scenario_studio.md file.

Please refer to https://docs.taipy.io/en/latest/manuals/userman/gui/pages for more details.
"""

# pages/scenario_studio.py

import taipy as tp
import taipy.gui.builder as tgb
from taipy.gui import notify

# ─── Callbacks ────────────────────────────────────────────────────────────────

def refresh_results_of_scenario(state):
    """
    Callback to refresh the results of the selected scenario.
    """
    with state as s:
        s.kpi_summary = s.selected_scenario.kpi_dn.read()
        s.model_metrics = s.selected_scenario.metrics_dn.read()
        s.feature_importance = s.selected_scenario.feature_importance_dn.read()


def change_scenario(state):
    """Reload widgets & results when a different scenario is selected."""
    with state as s:
        # ----- read DNs -----------------------------------------------------
        pg   = s.selected_scenario.price_grid_dn.read()     # _MapDict
        inv  = s.selected_scenario.inv_method_dn.read()     # _MapDict

        # ----- push plain-Python types into GUI bindings --------------------
        s.flash_date        = s.selected_scenario.flash_date_dn.read()
        s.price_grid_values = [float(pg["min"]), float(pg["max"])]  # list!
        s.price_grid_n      = int(pg.get("n", 0))

        s.unit_cost_factor  = float(
            s.selected_scenario.unit_cost_factor_dn.read()
        )

        # convert _MapDict to real dict so nested bindings behave
        s.inv_method        = dict(inv)

        s.synthetic_count   = int(
            s.selected_scenario.synthetic_count_dn.read()
        )
        s.marketing_boost   = int(
            s.selected_scenario.marketing_boost_dn.read()
        )
        s.model_type        = s.selected_scenario.model_type_dn.read()
        s.objective         = s.selected_scenario.objective_dn.read()

        # refresh KPI/model outputs
        refresh_results_of_scenario(s)


def on_submission_changed(state, submittable, details):
    if details["submission_status"] == "COMPLETED":
        print("Submission completed")
        refresh_results_of_scenario(state)
        notify(state, "s", "Submission completed")
    elif details["submission_status"] == "FAILED":
        notify(state, "error", "Submission failed")

def deactivate_scenario(state):
    """
    Callback to deactivate the scenario when creating a new one.
    """
    with state as s:
        s.active_scenario = False
        notify(s, "I", "Add settings to the new scenario before activating it.")

def add_tags_to_scenario(
        scenario, flash_date, price_grid_values, unit_cost_factor,
        inv_method, synthetic_count, marketing_boost, model_type, objective):
    """
    Add tags to the scenario based on the provided parameters.
    """
    tags = [
        f"flash_date:{flash_date}",
        f"price_grid:{price_grid_values[0]}-{price_grid_values[1]}",
        f"unit_cost_factor:{unit_cost_factor}",
        f"inv_method:{inv_method['method']}",
        f"synthetic_count:{synthetic_count}",
        f"marketing_boost:{marketing_boost}",
        f"model_type:{model_type}",
        f"objective:{objective}"
    ]
    scenario.tags = tags

    return scenario

def apply_settings(state):
    """
    Push the form values into the scenario and enable the submit widget.
    """
    with state as s:

        # 1) Build pickle-safe structures
        inv_method_clean = {
            "method": s.inv_method["method"],
            "s":      s.inv_method.get("s"),
            "Q":      s.inv_method.get("Q"),
            "R":      s.inv_method.get("R"),
            "S":      s.inv_method.get("S"),
            "initial_inventory": 50,
            "lead_time": {"type": "uniform_int", "low": 2, "high": 5}
        }

        # Price grid slider returns [min,max]; make sure n is present
        price_grid_clean = {
            "min": s.price_grid_values[0],
            "max": s.price_grid_values[1],
            "n":   s.price_grid_n,
        }

        # 2) Write into scenario DataNodes
        sc = s.selected_scenario
        sc.flash_date_dn.write(s.flash_date)
        sc.price_grid_dn.write(price_grid_clean)
        sc.unit_cost_factor_dn.write(s.unit_cost_factor)
        sc.inv_method_dn.write(inv_method_clean)
        sc.synthetic_count_dn.write(int(s.synthetic_count))
        sc.marketing_boost_dn.write(s.marketing_boost)
        sc.model_type_dn.write(s.model_type)
        sc.objective_dn.write(s.objective)

        # 3) Add tags for quick inspection
        sc.tags = [
            f"flash_date:{s.flash_date}",
            f"grid:{price_grid_clean['min']}-{price_grid_clean['max']}",
            f"unit_cost:{s.unit_cost_factor}",
            f"policy:{inv_method_clean['method']}",
            f"synthetic:{s.synthetic_count}",
            f"boost:{s.marketing_boost}",
            f"model:{s.model_type}",
            f"objective:{s.objective}"
        ]

        # 4) Tell the <|scenario|> widget it can run now
        s.active_scenario = True          # <-- this is the flag the widget uses
        notify(s, "s", "Settings applied — click ▶ to run the scenario")

# ─── Builder page ──────────────────────────────────────────────────────────────

with tgb.Page() as scenario_studio:

    # 1fr (sidebar) + 4fr (main); mobile → single-column
    with tgb.layout("1 4", columns__mobile="1"):

        # ── Sidebar: only the scenario selector ─────────────
        with tgb.part("sidebar"):
            tgb.text("### Select Scenario", mode="md")
            tgb.scenario_selector(
                "{selected_scenario}",
                on_change=change_scenario,
                on_creation=deactivate_scenario,
            )

        # ── Main: Create Scenario + Results ────────────────
        with tgb.part("main"):

            tgb.text("## Create Scenario", mode="md")

            # Row 1: Date | Price Grid | Grid Points | Cost Factor
            with tgb.layout("1 1 1 1", columns__mobile="2"):
                tgb.text("Flash Campaign Date", mode="md")
                tgb.date("{flash_date}")

                tgb.text("Price Grid (0.3×–2.0×)", mode="md")
                tgb.slider("{price_grid_values}", min=0.3, max=2.0, step=0.05)

                tgb.text("Grid Points", mode="md")
                tgb.input("{price_grid_n}")

                tgb.text("Unit Cost Factor", mode="md")
                tgb.slider("{unit_cost_factor}", min=0, max=1, step=0.01)

            # Row 2: Policy & its (s,Q) fields | R
            with tgb.layout("1 1 1 1", columns__mobile="2"):
                tgb.text("Inventory Policy", mode="md")
                tgb.selector("{inv_method['method']}", lov=["s,Q", "R,S"])

                tgb.text("Reorder Point (s)", mode="md")
                tgb.number(
                    "{inv_method['s']}",
                    if_="inv_method['method']=='s,Q'",
                )

                tgb.text("Order Qty (Q)", mode="md")
                tgb.number(
                    "{inv_method['Q']}",
                    if_="inv_method['method']=='s,Q'",
                )

                tgb.text("Review Period (R)", mode="md")
                tgb.number(
                    "{inv_method['R']}",
                    if_="inv_method['method']=='R,S'",
                )

            # Row 3: S | Synthetic | Marketing | Model Type
            with tgb.layout("1 1 1 1", columns__mobile="2"):
                tgb.text("Order-Up-To (S)", mode="md")
                tgb.number(
                    "{inv_method['S']}",
                    if_="inv_method['method']=='R,S'",
                )

                tgb.text("Synthetic Orders Count", mode="md")
                tgb.number("{synthetic_count}")

                tgb.text("Marketing Boost (%)", mode="md")
                tgb.slider("{marketing_boost}", min=1, max=100, step=1)

                tgb.text("Model Type", mode="md")
                tgb.selector("{model_type}", lov=["xgboost", "linear"])

            # Row 4: Objective → then Apply + hidden scenario submit
            with tgb.layout("1 1 1 1", columns__mobile="2"):
                tgb.text("Objective", mode="md")
                tgb.toggle("{objective}", lov=["profit", "revenue"])

            tgb.button(
                "Apply Settings",
                on_action=apply_settings,
                class_name="fullwidth mb2",
            )

            # hidden behind the scenes scenario submit
            tgb.scenario(
                "{selected_scenario}",
                show_sequences=False,
                on_submission_change=on_submission_changed,
                active="{active_scenario}",
            )

            # ── Scenario Results ────────────────────────────────
            tgb.text("## Scenario Results", mode="md")
            tgb.table("{kpi_summary}", expanded=False)