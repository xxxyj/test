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

# build tf.data pipeline
batch_size = config.train["batch_size"]


def make_dataset(inputs):
    """Create an infinite tf.data.Dataset from the list of numpy arrays.

    The previous implementation used a Python generator which prevented
    TensorFlow from overlapping the data preparation with model execution. By
    converting the generator into a ``tf.data`` pipeline with prefetching we
    allow for proper pipelining and better performance.
    """

    def gen():
        while True:
            random.shuffle(inputs)
            for tensors in inputs:
                sequence = Sequence(*tensors, batch_size)
                for i in range(sequence._num_batches):
                    yield sequence[i]

    output_signature = (
        {
            "rec_input": tf.TensorSpec(shape=(None, None, None), dtype=tf.int32),
            "eval_input": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    return (
        tf.data.Dataset.from_generator(gen, output_signature=output_signature)
        .prefetch(tf.data.AUTOTUNE)
    )


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

train_ds = make_dataset(train_data)
val_ds = make_dataset(val_data)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.train["epochs"],
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=train["shots"]
    * len(train["rounds"])
    * len(train["states"])
    // batch_size,
    validation_steps=val["shots"]
    * len(val["rounds"])
    * len(val["states"])
    // batch_size,
)
model.save(config.checkpoint_dir / "final_weights.keras")

