"""
This is the main entry point for the Flash Pricer application.
It initializes the Taipy Orchestrator and GUI, and sets up the pages for the application.
"""


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

