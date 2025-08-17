import pathlib
import os

from qrennd import Config, Layout, get_model

from lib.evaluation import evaluate_model

# Parameters
EXP_NAME = "yuejie_data"
MODEL_NAME = "20250516-172801_slru_lstm24x2_eval24_b64_dr0-10_lr0-001"
TEST_DATASET = ["test"]

LAYOUT_NAME = "rep_code_d3_bZ.yaml"

OVERWRITE = True

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

CONFIG_FILE = MODEL_DIR / "config.yaml"
if not CONFIG_FILE.exists():
    raise ValueError(f"Config file does not exist: {CONFIG_FILE}")

LAYOUT_FILE = DATA_DIR / EXP_NAME / "config" / LAYOUT_NAME
if not LAYOUT_FILE.exists():
    raise ValueError(f"Layout file does not exist: {LAYOUT_FILE}")


# evaluate the NN
layout = Layout.from_yaml(LAYOUT_FILE)
config = Config.from_yaml(
    filepath=CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)
if not isinstance(TEST_DATASET, list):
    TEST_DATASET = [TEST_DATASET]

# if results have not been stored, evaluate model
for test_dataset in TEST_DATASET:
    print("")

    NAME = f"{test_dataset}.nc"
    if (MODEL_DIR / NAME).exists() and (not OVERWRITE):
        print("Model already evaluated!")
        print("output_dir=", OUTPUT_DIR)
        print("exp_name=", EXP_NAME)
        print("run_name=", MODEL_NAME)
        print("test_data=", test_dataset)
        continue

    print("Evaluating model...")

    anc_qubits = layout.get_qubits(role="anc")
    num_anc = len(anc_qubits)

    rec_features = 0
    eval_features = num_anc
    if "outcomes" in config.dataset["input_names"]:
        rec_features += num_anc
    if "binary_outcomes" in config.dataset["input_names"]:
        rec_features += num_anc
    if "defects" in config.dataset["input_names"]:
        rec_features += num_anc
    if "leakage_flags" in config.dataset["input_names"]:
        rec_features += num_anc

    model = get_model(
        rec_features=rec_features,
        eval_features=eval_features,
        config=config,
    )

    model.load_weights(MODEL_DIR / "checkpoint/weights.hdf5")
    log_fid = evaluate_model(model, config, layout, test_dataset)
    log_fid.to_netcdf(path=MODEL_DIR / NAME)

    print("Evaluation completed!")
    print("output_dir=", OUTPUT_DIR)
    print("exp_name=", EXP_NAME)
    print("run_name=", MODEL_NAME)
    print("test_data=", test_dataset)
