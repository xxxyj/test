import pathlib
import numpy as np
import xarray as xr

from lib.layouts import rep_code
from lib.preprocessing import to_inputs

DC_DATA_DIR = "dc_data_rnd_flp"
QRENND_DATA_DIR = "qrend_data/yuejie_data_rndflp"

DISTANCE = 3
BASIS = "Z"

TRAIN = {
    "dir_name": "train",
    "file_name": "lru_round_{num_rounds}_leak_3_state_{state}.txt",
    "rounds": [1,  2,  3,  4,  5,  7,  8, 11, 14, 17, 20, 26, 32, 41],
    "states": list(range(2**DISTANCE)),
}
VAL_FRACTION = 0.125
TEST = {
    "dir_name": "test",
    "file_name": "lru_round_{num_rounds}_leak_3_state_{state}.txt",
    "rounds": [1,  2,  3,  4,  5,  7,  8, 10, 13, 16, 21, 26, 33, 42, 53, 68, 87, 108],
    "states": list(range(2**DISTANCE)),
}

QRENND_DIR_NAME = "repcode_d{distance}_r{num_rounds}_s{state}_b{basis}"
QRENND_FILE_NAME = "measurements.nc"

state_to_bitstring = lambda x: f"{x:0{DISTANCE}b}"[::-1]  # D0,D1,D2,... ordering


###################

DC_DATA_DIR = pathlib.Path(DC_DATA_DIR)
if not DC_DATA_DIR.exists():
    raise ValueError(f"{DC_DATA_DIR} is not an existing directory.")

QRENND_DATA_DIR = pathlib.Path(QRENND_DATA_DIR)
QRENND_DATA_DIR.mkdir(exist_ok=True, parents=True)

QRENND_TRAIN_DIR = QRENND_DATA_DIR / "train"
QRENND_TRAIN_DIR.mkdir(exist_ok=True, parents=True)
QRENND_VAL_DIR = QRENND_DATA_DIR / "val"
QRENND_VAL_DIR.mkdir(exist_ok=True, parents=True)
QRENND_TEST_DIR = QRENND_DATA_DIR / "test"
QRENND_TEST_DIR.mkdir(exist_ok=True, parents=True)

QRENND_CONFIG_DIR = QRENND_DATA_DIR / "config"
QRENND_CONFIG_DIR.mkdir(exist_ok=True, parents=True)
layout = rep_code(DISTANCE, BASIS)
layout.to_yaml(QRENND_CONFIG_DIR / f"rep_code_d{DISTANCE}_b{BASIS}.yaml")

# process 'Train' data and split it to training and validation
for num_rounds in TRAIN["rounds"]:
    for state in TRAIN["states"]:
        # create file names and corresponding direcotires
        dc_file_name = TRAIN["file_name"].format(num_rounds=num_rounds, state=state)
        qrennd_dir_name = QRENND_DIR_NAME.format(
            distance=DISTANCE,
            num_rounds=num_rounds,
            state=state_to_bitstring(state),
            basis=BASIS,
        )
        (QRENND_TRAIN_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)
        (QRENND_VAL_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)

        # process DiCarlo data
        with open(DC_DATA_DIR / TRAIN["dir_name"] / dc_file_name, "r") as f:
            raw_data = f.read()

        outcomes = [
            list(map(int, line.split("\t")))
            for line in raw_data.split("\n")
            if line != ""
        ]
        outcomes = np.array(outcomes, dtype=int)

        num_shots, num_meas = outcomes.shape
        assert num_meas == (DISTANCE - 1) * num_rounds + DISTANCE

        data_meas = outcomes[:, -DISTANCE:]
        anc_meas = outcomes[:, :-DISTANCE]
        anc_meas = anc_meas.reshape(num_shots, DISTANCE - 1, num_rounds)
        anc_meas = anc_meas.swapaxes(1, 2)
        # anc_meas axes = (num_shots, num_rounds, anc_qubit)
        assert data_meas.max() <= 1
        data_meas = data_meas.astype(bool)

        # ideal measurements for a single shot
        bitstring = np.array(list(map(int, state_to_bitstring(state))), dtype=bool)
        projected_bitstring = np.array(
            [i ^ j for i, j in zip(bitstring[:-1], bitstring[1:])], dtype=bool
        )
        ideal_anc_meas = (
            np.zeros((num_rounds, DISTANCE - 1), dtype=bool) ^ projected_bitstring
        )
        ideal_anc_meas[1::2] ^= projected_bitstring  # no resets in ancillas
        ideal_data_meas = np.zeros_like(DISTANCE, dtype=bool) ^ bitstring
        if num_rounds == 0:
            pass
        if num_rounds % 2 == 0:
            ideal_data_meas ^= True  # echo gates

        # split dataset into training and eval
        idx = int(num_shots * (1 - VAL_FRACTION))
        train_data_meas, val_data_meas = data_meas[:idx], data_meas[idx:]
        train_anc_meas, val_anc_meas = anc_meas[:idx], anc_meas[idx:]

        train_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), train_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                data_meas=(("shot", "data_qubit"), train_data_meas),
                ideal_data_meas=(("data_qubit"), ideal_data_meas),
            ),
            coords=dict(
                shot=list(range(idx)),
                qec_round=list(range(1, num_rounds + 1)),
                data_qubit=[f"D{i}" for i in range(DISTANCE)],
                anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        train_ds.to_netcdf(QRENND_TRAIN_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        val_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), val_anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                data_meas=(("shot", "data_qubit"), val_data_meas),
                ideal_data_meas=(("data_qubit"), ideal_data_meas),
            ),
            coords=dict(
                shot=list(range(idx, num_shots)),
                qec_round=list(range(1, num_rounds + 1)),
                data_qubit=[f"D{i}" for i in range(DISTANCE)],
                anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        val_ds.to_netcdf(QRENND_VAL_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        # display useful analysis
        print(qrennd_dir_name)
        print(f"total number of training+val shots: {num_shots}")
        for state in [0, 1, 2]:
            fraction = np.average(train_ds.anc_meas.values == state)
            print(f"{state}s fraction (train) = {fraction:0.3f}")

        proj_matrix = layout.projection_matrix(f"{BASIS.lower()}_type")
        defects, final_defects, log_flips = to_inputs(
            train_ds, proj_matrix, input_names=["defects"]
        )
        defects = defects[0].mean(dim="shot")
        final_defects = final_defects.mean(dim="shot")
        defects = np.concatenate(
            [defects.values, final_defects.values.reshape(1, -1)], axis=0
        )
        print(f"defect rates: {np.average(defects):0.4f} +/- {np.std(defects):0.4f}")
        print(f"logical flips: {log_flips.mean():0.4f}")
        print("")


# process 'Test' data
for num_rounds in TEST["rounds"]:
    for state in TEST["states"]:
        # create file names and corresponding direcotires
        dc_file_name = TEST["file_name"].format(num_rounds=num_rounds, state=state)
        qrennd_dir_name = QRENND_DIR_NAME.format(
            distance=DISTANCE,
            num_rounds=num_rounds,
            state=state_to_bitstring(state),
            basis=BASIS,
        )
        (QRENND_TEST_DIR / qrennd_dir_name).mkdir(exist_ok=True, parents=True)

        # process DiCarlo data
        with open(DC_DATA_DIR / TEST["dir_name"] / dc_file_name, "r") as f:
            raw_data = f.read()

        outcomes = [
            list(map(int, line.split("\t")))
            for line in raw_data.split("\n")
            if line != ""
        ]
        outcomes = np.array(outcomes, dtype=int)

        num_shots, num_meas = outcomes.shape
        assert num_meas == (DISTANCE - 1) * num_rounds + DISTANCE

        data_meas = outcomes[:, -DISTANCE:]
        anc_meas = outcomes[:, :-DISTANCE]
        anc_meas = anc_meas.reshape(num_shots, DISTANCE - 1, num_rounds)
        anc_meas = anc_meas.swapaxes(1, 2)
        # anc_meas axes = (num_shots, num_rounds, anc_qubit)
        assert data_meas.max() <= 1
        data_meas = data_meas.astype(bool)

        # ideal measurements for a single shot
        bitstring = np.array(list(map(int, state_to_bitstring(state))), dtype=bool)
        projected_bitstring = np.array(
            [i ^ j for i, j in zip(bitstring[:-1], bitstring[1:])], dtype=bool
        )
        ideal_anc_meas = (
            np.zeros((num_rounds, DISTANCE - 1), dtype=bool) ^ projected_bitstring
        )
        ideal_anc_meas[1::2] ^= projected_bitstring  # no resets in ancillas
        ideal_data_meas = np.zeros_like(DISTANCE, dtype=bool) ^ bitstring
        if num_rounds == 0:
            pass
        if num_rounds % 2 == 0:
            ideal_data_meas ^= True  # echo gates

        test_ds = xr.Dataset(
            data_vars=dict(
                anc_meas=(("shot", "qec_round", "anc_qubit"), anc_meas),
                ideal_anc_meas=(("qec_round", "anc_qubit"), ideal_anc_meas),
                data_meas=(("shot", "data_qubit"), data_meas),
                ideal_data_meas=(("data_qubit"), ideal_data_meas),
            ),
            coords=dict(
                shot=list(range(num_shots)),
                qec_round=list(range(1, num_rounds + 1)),
                data_qubit=[f"D{i}" for i in range(DISTANCE)],
                anc_qubit=[f"A{i}" for i in range(DISTANCE - 1)],
                meas_reset=False,
                basis=BASIS,
            ),
        )
        test_ds.to_netcdf(QRENND_TEST_DIR / qrennd_dir_name / QRENND_FILE_NAME)

        # display useful analysis
        print(qrennd_dir_name)
        print(f"total number of testing shots: {num_shots}")
        for state in [0, 1, 2]:
            fraction = np.average(test_ds.anc_meas.values == state)
            print(f"{state}s fraction (test) = {fraction:0.3f}")

        proj_matrix = layout.projection_matrix(f"{BASIS.lower()}_type")
        defects, final_defects, log_flips = to_inputs(
            test_ds, proj_matrix, input_names=["defects"]
        )
        defects = defects[0].mean(dim="shot")
        final_defects = final_defects.mean(dim="shot")
        defects = np.concatenate(
            [defects.values, final_defects.values.reshape(1, -1)], axis=0
        )
        print(f"defect rates: {np.average(defects):0.4f} +/- {np.std(defects):0.4f}")
        print(f"logical flips: {log_flips.mean():0.4f}")
        print("")
