import pathlib
import os

import xarray as xr
from matplotlib import pyplot as plt

from qec_util.performance import (
    LogicalErrorProbDecayModel,
    lmfit_par_to_ufloat,
    logical_error_prob_decay,
)

# Parameters
EXP_NAME = "yuejie_data"
TEST_DATASET = ["test", "val"]

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

MODEL_NAMES = os.listdir(OUTPUT_DIR / EXP_NAME)
MODEL_NAMES = [n for n in MODEL_NAMES if n not in ["__old", ".DS_Store"]]

for MODEL_NAME in MODEL_NAMES:
    MODEL_DIR = OUTPUT_DIR / EXP_NAME / MODEL_NAME
    if not MODEL_DIR.exists():
        raise ValueError(f"Model directory does not exist: {MODEL_DIR}")

    # plot evaluation of the NN
    if not isinstance(TEST_DATASET, list):
        TEST_DATASET = [TEST_DATASET]

    # if results have not been stored, evaluate model
    for test_dataset in TEST_DATASET:
        NAME = f"{test_dataset}.nc"
        if not (MODEL_DIR / NAME).exists():
            print(f"No dataset found for {MODEL_DIR / NAME}")
            continue

        log_fid = xr.load_dataset(MODEL_DIR / NAME)

        fig, ax = plt.subplots()

        log_prob = log_fid.log_errors.transpose("state", "qec_round", "shot").values
        states = log_fid.state.values
        log_probs = log_prob.mean(axis=(2,))
        for state, log_prob in zip(states, log_probs):
            ax.plot(log_fid.qec_round, 1 - log_prob, ".", label=state)

        """
        rounds = log_fid.qec_round.values
        log_prob = log_prob[rounds >= 5]
        rounds = rounds[rounds >= 5]
        model = LogicalErrorProbDecayModel()
        # guess = model.guess(log_prob, x=rounds)
        fit = model.fit(log_prob, rounds)

        error_rate = lmfit_par_to_ufloat(fit.params["error_rate"]).nominal_value
        qec_offset = lmfit_par_to_ufloat(fit.params["qec_offset"]).nominal_value

        log_prob_fit = logical_error_prob_decay(rounds, error_rate, qec_offset)
        ax.plot(
            rounds,
            1 - log_prob_fit,
            "-",
            color="black",
            label=f"$\\epsilon_L = {error_rate*100:0.3f}$%",
        )
        """

        ax.set_xlim(xmin=0)
        ax.set_ylim(0.5, 1)
        ax.set_xlabel("QEC round")
        ax.set_ylabel("logical fidelity, $1 - p_L$")

        ax.legend(loc="lower left")

        fig.tight_layout()
        fig.savefig(MODEL_DIR / f"log_fid_{test_dataset}_states.pdf", format="pdf")
        fig.savefig(MODEL_DIR / f"log_fid_{test_dataset}_states.png", format="png")

        plt.cla()
        plt.close()
