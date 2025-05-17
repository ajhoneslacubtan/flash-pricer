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
# main.py

# main.py
# ------------------------------------------------------------
import taipy as tp
from taipy.gui import Gui
from taipy import Orchestrator
from pages import *
from configuration.config import scenario_cfg

# Define the pages for the GUI
pages = {
    "/": root_page,
    "dashboard": dashboard,
    "scenario_studio": scenario_studio,
    "data_explorer": data_explorer,
    "model_insights": model_insights,
    "run_history": run_history
}

if __name__ == "__main__":
    # Start the Orchestrator
    orchestrator = Orchestrator()
    orchestrator.run()

    # Launch the GUI
    gui = Gui(pages=pages)
    gui.run(title="Flash Pricer")

