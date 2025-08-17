import numpy as np
import xarray as xr

from .preprocessing import to_inputs, to_model_input


def evaluate_model(model, config, layout, dataset_name="test"):
    rounds = config.dataset[dataset_name]["rounds"]
    states = config.dataset[dataset_name]["states"]
    shots = config.dataset[dataset_name]["shots"]
    experiment_name = config.dataset["folder_format_name"]
    input_names = config.dataset["input_names"]
    data_type = int if "outcomes" in input_names else bool

    basis = config.dataset["basis"]
    stab_type = f"{basis.lower()}_type"

    dataset_dir = config.experiment_dir / dataset_name
    dataset_params = config.dataset[dataset_name]
    dataset_params["distance"] = config.dataset["distance"]
    for name in ["states", "rounds", "shots"]:
        dataset_params.pop(name)

    # avoid error when num_rounds=0
    if 0 in rounds:
        rounds.pop(rounds.index(0))

    # compute num_shots if not specified (None)
    experiment = experiment_name.format(
        basis=basis,
        state=states[0],
        shots=shots,
        num_rounds=rounds[0],
        **dataset_params,
    )
    dataset = xr.open_dataset(dataset_dir / experiment / "measurements.nc")
    shots = len(dataset.shot)

    log_errors = np.zeros((len(rounds), len(states), shots))
    for i, num_rounds in enumerate(rounds):
        for j, state in enumerate(states):
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                shots=shots,
                num_rounds=num_rounds,
                **dataset_params,
            )
            dataset = xr.open_dataset(dataset_dir / experiment / "measurements.nc")
            proj_matrix = layout.projection_matrix(stab_type)

            # Convert to desired input
            dataset = to_inputs(dataset, proj_matrix, input_names=input_names)

            # Process for keras.model input
            dataset = to_model_input(*dataset, data_type=data_type)

            print(f"QEC = {num_rounds} | state = {state}", end="\r")
            prediction = model.predict((dataset[0], dataset[1]), verbose=0)
            prediction = (prediction[0] > 0.5).flatten()
            errors = prediction != dataset[2]
            log_errors[i, j] = errors
            print(
                f"QEC = {num_rounds} | state = {state} | avg_errors = {np.average(errors):.4f}",
                end="\r",
            )

    log_fid = xr.Dataset(
        data_vars=dict(log_errors=(["qec_round", "state", "shot"], log_errors)),
        coords=dict(qec_round=rounds, state=states, shot=list(range(shots))),
    )

    return log_fid
