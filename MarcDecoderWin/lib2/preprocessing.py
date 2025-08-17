from typing import List, Tuple, Union
from copy import deepcopy
import numpy as np
import xarray as xr

from qrennd.datasets.preprocessing import get_syndromes, get_defects, get_final_defects


def to_inputs(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    input_names: List[str],
) -> Tuple[List[xr.DataArray], xr.DataArray, xr.DataArray]:
    """
    Preprocess dataset to generate defect inputs, leakage flags,
    and the logical errors.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas (int): [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas (int): [qec_cycle, anc_qubit]
        - data_meas (bool): [shot, data_qubit]
        - idea_data_meas (bool): [data_qubit]
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    input_names
        List of input names to be given to the NN.
        The available input names are:
            "defects", "outcomes", "leakage_flags", "binary_outcomes"
        Note that these refer to the QEC cycles, not the final defects (obtained
        from the data qubits). For the final defects, only the defects are given.
    """
    anc_meas = dataset.anc_meas
    leakage_flags = xr.where(anc_meas == 2, True, False).astype(bool)

    anc_meas_bin = xr.where(anc_meas == 2, 1, anc_meas).astype(bool)
    anc_flips = anc_meas_bin ^ dataset.ideal_anc_meas
    syndromes = get_syndromes(anc_flips)
    defects = get_defects(syndromes)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    proj_syndrome = (data_flips @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    log_errors = data_flips.sum(dim="data_qubit") % 2

    available_inputs = set(["defects", "outcomes", "leakage_flags", "binary_outcomes"])
    if available_inputs < set(input_names):
        raise ValueError(
            f"The available inputs are '{available_inputs}', "
            f"but '{input_names}' was given."
        )
    rec_inputs = []
    if "defects" in input_names:
        rec_inputs.append(defects)
    if "outcomes" in input_names:
        rec_inputs.append(anc_meas)
    if "binary_outcomes" in input_names:
        rec_inputs.append(anc_meas_bin)
    if "leakage_flags" in input_names:
        rec_inputs.append(leakage_flags)

    # close dataset to avoid memory issues
    dataset.close()

    return rec_inputs, final_defects, log_errors


def add_classical_meas_noise(
    dataset: xr.Dataset, noise_params: Union[dict, None]
) -> xr.Dataset:
    if noise_params is None:
        return dataset

    noiseless_anc_meas = deepcopy(dataset.anc_meas)

    for state in [0, 1, 2]:
        noisy_anc_meas = np.random.choice(
            [0, 1, 2], size=dataset.anc_meas.values.shape, p=noise_params[state]
        )
        dataset["anc_meas"] = xr.where(
            noiseless_anc_meas == state, noisy_anc_meas, dataset.anc_meas
        )

    return dataset


def to_model_input(
    rec_inputs: xr.DataArray,
    eval_inputs: xr.DataArray,
    log_errors: xr.DataArray,
    data_type: type = bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(rec_inputs, list):
        rec_tensor = [r.values.astype(np.int8) for r in rec_inputs]
        rec_tensor = np.concatenate(rec_tensor, axis=2, dtype=np.int8)
        # close dataarrays to avoid memory issues
        for r in rec_inputs:
            r.close()
    else:
        rec_tensor = rec_inputs.values.astype(data_type)
        # close dataarrays to avoid memory issues
        rec_inputs.close()

    if isinstance(eval_inputs, list):
        eval_tensor = [r.values.astype(np.int8) for r in eval_inputs]
        eval_tensor = np.concatenate(eval_tensor, axis=1, dtype=np.int8)
        # close dataarrays to avoid memory issues
        for r in eval_inputs:
            r.close()
    else:
        eval_tensor = eval_inputs.values.astype(data_type)
        # close dataarrays to avoid memory issues
        eval_inputs.close()

    error_tensor = log_errors.values.astype(data_type)
    # close dataarrays to avoid memory issues
    log_errors.close()

    return rec_tensor, eval_tensor, error_tensor
