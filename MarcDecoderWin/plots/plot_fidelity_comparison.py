import pathlib
import os
from copy import deepcopy

import xarray as xr
from matplotlib import pyplot as plt

from qec_util.performance import (
    LogicalErrorProbDecayModel,
    lmfit_par_to_ufloat,
    logical_error_prob_decay,
)

# Parameters
EXP_NAME = "20250408_repcode_d3_simulated"
MODEL_NAMES = {
    "20250409-131401_nslru_lstm16x2_eval16_b64_dr0-10_lr0-001": {
        "color": "lightblue",
        "linestyle": "-",
    },
    "20250410-185417_slru_lstm16x2_eval16_b64_dr0-10_lr0-001": {
        "color": "orange",
        "linestyle": "-",
    },
    "20250417-004643_slru_lstm24x2_eval24_b64_dr0-10_lr0-001": {
        "color": "darkorange",
        "linestyle": "-",
    },
}
TEST_DATASET = ["test"]

USERNAME = os.environ.get("USER")
HOME_DIR = pathlib.Path(f"/scratch/{USERNAME}")
HOME_DIR = pathlib.Path("/Users/marcserraperal/marc")
PARENT_DIR = HOME_DIR / "projects" / "20250402-signaling_lru_decoding_dicarlo_lab"

DATA_DIR = PARENT_DIR / "data"
OUTPUT_DIR = PARENT_DIR / "output"

####################################

if not DATA_DIR.exists():
    raise ValueError(f"Data directory does not exist: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    raise ValueError(f"Output directory does not exist: {OUTPUT_DIR}")

fig, ax = plt.subplots()

for MODEL_NAME in MODEL_NAMES:
    MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
    if not MODEL_DIR.exists():
        raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

    if not isinstance(TEST_DATASET, list):
        TEST_DATASET = [TEST_DATASET]

    for test_dataset in TEST_DATASET:
        NAME = f"{test_dataset}.nc"
        if not (MODEL_DIR / NAME).exists():
            print(f"No dataset found for {MODEL_DIR / NAME}")
            continue

        log_fid = xr.load_dataset(MODEL_DIR / NAME)

        log_prob = log_fid.log_errors.transpose("qec_round", "state", "shot").values
        log_prob = log_prob.mean(axis=(1, 2))
        points_kargs = deepcopy(MODEL_NAMES[MODEL_NAME])
        points_kargs["linestyle"] = "none"
        points_kargs["marker"] = "."
        ax.plot(log_fid.qec_round, 1 - log_prob, **points_kargs)

        rounds = log_fid.qec_round.values
        log_prob = log_prob[rounds >= 5]
        rounds = rounds[rounds >= 5]
        model = LogicalErrorProbDecayModel()
        # guess = model.guess(log_prob, x=rounds)
        fit = model.fit(log_prob, rounds)

        error_rate = lmfit_par_to_ufloat(fit.params["error_rate"]).nominal_value
        qec_offset = lmfit_par_to_ufloat(fit.params["qec_offset"]).nominal_value

        log_prob_fit = logical_error_prob_decay(rounds, error_rate, qec_offset)
        lines_kargs = deepcopy(MODEL_NAMES[MODEL_NAME])
        lines_kargs["marker"] = "none"
        ax.plot(
            rounds,
            1 - log_prob_fit,
            label=f"$\\epsilon_L = {error_rate*100:0.3f}$%",
            **lines_kargs,
        )

ax.set_xlim(xmin=0)
ax.set_ylim(0.5, 1)
ax.set_xlabel("QEC round")
ax.set_ylabel("logical fidelity, $1 - p_L$")

ax.set_title("\n".join(MODEL_NAMES))

ax.legend(loc="lower left")

fig.tight_layout()
fig.savefig("comparison.pdf", format="pdf")
fig.savefig("comparison.png", format="png")

plt.show()
