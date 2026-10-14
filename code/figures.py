"""Draws Figures 3-5 from out/paper/ (written by paper_numbers.py) into out/figures/."""
import runpy
from pathlib import Path

for name in ("fig_main_effects", "fig_scenarios", "fig_environment"):
    runpy.run_path(str(Path(__file__).resolve().parent / "figures" / f"{name}.py"), run_name="__main__")
