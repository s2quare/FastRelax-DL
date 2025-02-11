"""
Exposes common paths useful for manipulating datasets and generating figures.

Author: florian.birk@tuebingen.mpg.de (Florian Birk), February 2024
"""
from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[1].absolute()

# Path for simulation and training configuration files
cfgs = root / "cfgs"

# Absolute path to the data folder 
data = root / "data"

# Absolute path to best trained models
dnn_models = root / "models"

# Absolute path to the `src/scripts`
scripts = root / "scripts"

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = root / "figures"
