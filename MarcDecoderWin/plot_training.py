import pathlib
import os

import pandas as pd
from matplotlib import pyplot as plt

# Parameters
EXP_NAME = "yuejie_data"
MODEL_NAME = "20250516-172801_slru_lstm24x2_eval24_b64_dr0-10_lr0-001"

USERNAME = os.environ.get("USER")
HOME_DIR = pathlib.Path(f"/scratch/{USERNAME}")
HOME_DIR = pathlib.Path("/Users/marcserraperal/marc")
PARENT_DIR = HOME_DIR / "projects" / "20250402-signaling_lru_decoding_dicarlo_lab"

DATA_DIR = pathlib.Path("qrend_data")
OUTPUT_DIR = pathlib.Path("output")

####################################


if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
if not MODEL_DIR.exists():
    raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

LOG_FILE = MODEL_DIR / "logs" / "training.log"
if not LOG_FILE.exists():
    raise ValueError(f"Log file does not exist: {LOG_FILE}")

# analyize the training
dataframe = pd.read_csv(LOG_FILE, delimiter="\t")
try:
    dataframe.epoch
except:
    dataframe = pd.read_csv(LOG_FILE, delimiter=",")


METRICS = ("loss", "main_output_accuracy")
for metric in METRICS:
    fig, ax = plt.subplots()

    ax.plot(dataframe.epoch, dataframe[metric], ".-", color="blue", label="training")
    ax.plot(
        dataframe.epoch,
        dataframe["val_" + metric],
        ".-",
        color="orange",
        label="validation",
    )

    ax.legend(frameon=False)
    ax.set_xlabel("epochs")
    ax.set_ylabel(metric.replace("_", " "))

    fig.tight_layout()
    fig.savefig(MODEL_DIR / f"{metric}.pdf", format="pdf")
    fig.savefig(MODEL_DIR / f"{metric}.png", format="png")

plt.show()
