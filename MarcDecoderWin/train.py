import os
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


def make_dataset(rec_input, eval_input, labels, training=False):
    """Construct a ``tf.data.Dataset`` from numpy arrays.

    Using ``from_tensor_slices`` removes Python overhead from the input
    pipeline, enabling TensorFlow to better overlap input processing with GPU
    execution. When ``training`` is ``True`` the dataset is shuffled and
    repeated to provide an infinite stream of data.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        ({"rec_input": rec_input, "eval_input": eval_input}, labels)
    )
    dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(len(labels)).repeat()
    dataset = dataset.batch(batch_size)

    # Prefetch to GPU if available to hide host-to-device transfer latency
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        dataset = dataset.apply(
            tf.data.experimental.copy_to_device("/GPU:0")
        ).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


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
train_rec, train_eval, train_labels = train_data
val_rec, val_eval, val_labels = val_data

train_ds = make_dataset(train_rec, train_eval, train_labels, training=True)
val_ds = make_dataset(val_rec, val_eval, val_labels)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config.train["epochs"],
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=train_rec.shape[0] // batch_size,
    validation_steps=val_rec.shape[0] // batch_size,
)
model.save_weights(config.checkpoint_dir / "final_weights.h5")

