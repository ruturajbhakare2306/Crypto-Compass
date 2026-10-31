import pandas as pd
from pathlib import Path

def load_file():

    project = Path(__file__).resolve().parent.parent
    data = project / "data" / "output.csv"
    df = pd.read_csv(data)
    return df