import os
import gc
import pathlib
import random
import numpy as np
import tensorflow as tf

from qrennd import (
    Layout,
    Config,
    get_model,
)

from lib.util import load_datasets
from lib.callbacks import get_callbacks
from lib.sequences import Sequence
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# https://github.com/tensorflow/tensorflow/issues/35100
# def fix_gpu():
#     config = ConfigProto()
#     config.gpu_options.allow_growth = True
#     session = InteractiveSession(config=config)
#
#
# fix_gpu()

tf.get_logger().setLevel("DEBUG")

######################################

# Parameters
LAYOUT_FILE = "rep_code_d3_bZ.yaml"
CONFIG_FILE = "config_slru.yaml"

USERNAME = os.environ.get("USER")
HOME_DIR = pathlib.Path(f"/scratch/{USERNAME}")
HOME_DIR = pathlib.Path("/Users/marcserraperal/marc")
PARENT_DIR = HOME_DIR / "projects" / "20250402-signaling_lru_decoding_dicarlo_lab"

DATA_DIR = pathlib.Path("qrend_data")
OUTPUT_DIR = pathlib.Path("output")

###################################

# Load setup objects
CONFIG_DIR = pathlib.Path.cwd() / "configs"
config = Config.from_yaml(
    filepath=CONFIG_DIR / CONFIG_FILE,
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
)
config.log_dir.mkdir(exist_ok=True, parents=True)
config.checkpoint_dir.mkdir(exist_ok=True, parents=True)

# LAYOUT_DIR = DATA_DIR / config.experiment /   "config"
LAYOUT_DIR = DATA_DIR / config.experiment / "config"
layout = Layout.from_yaml(LAYOUT_DIR / LAYOUT_FILE)

# set random seed for tensorflow, numpy and python
# ensure that the new seed is stored in config for reproducible results
if config.seed is None:
    config.seed = int(np.random.get_state()[1][0])
random.seed(config.seed)
np.random.seed(config.seed)
tf.random.set_seed(config.seed)

config.to_yaml(config.run_dir / "config.yaml")

# load datasets
# load datasets
print("loading validation data...")
val_data = load_datasets(config=config, layout=layout, dataset_name="val")
print("loading training data...")
train_data = load_datasets(config=config, layout=layout, dataset_name="train")
print("completed")

# this is for model.fit to know that the num_rounds coordinate is not fixed
batch_size = config.train["batch_size"]
tensor1, tensor2 = train_data[0], train_data[-1]
seq1, seq2 = Sequence(*tensor1, batch_size), Sequence(*tensor2, batch_size)
first_batch, second_batch = seq1[0], seq2[0]


def infinite_gen(inputs):
    while True:
        random.shuffle(train_data)
        sequences = (Sequence(*tensors, batch_size) for tensors in inputs)
        # this is for model.fit to know that the num_rounds coordinate is not fixed
        yield first_batch
        yield second_batch

        for k, sequence in enumerate(sequences):
            # cannot do 'yield from sequence' because it has no end!
            for i in range(sequence._num_batches):
                yield sequence[i]


# load model
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

model.summary()

callbacks = get_callbacks(config)


# train model
train = config.dataset["train"]
val = config.dataset["val"]
batch_size = config.train["batch_size"]
history = model.fit(
    infinite_gen(train_data),
    validation_data=infinite_gen(val_data),
    # batch_size=config.train["batch_size"],
    epochs=config.train["epochs"],
    callbacks=callbacks,
    # shuffle=True,
    verbose=1,
    steps_per_epoch=train["shots"]
    * len(train["rounds"])
    * len(train["states"])
    // batch_size
    + 2,  # +2 is for model.fit to know that the num_rounds coordinate is not fixed
    validation_steps=val["shots"]
    * len(val["rounds"])
    * len(val["states"])
    // batch_size
    + 2,  # +2 is for model.fit to know that the num_rounds coordinate is not fixed
)
model.save(config.checkpoint_dir / "final_weights.keras")

