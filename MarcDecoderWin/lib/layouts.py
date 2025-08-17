from collections import defaultdict
from qrennd import Layout


def rep_code(
    distance: int,
    basis: str,
    logical_qubit_label: str = "L0",
    init_point=(1, 1),
    init_data_qubit_id: int = 0,
    init_anc_qubit_id: int = 0,
    init_ind: int = 0,
) -> Layout:
    """Generates a repetition code layout.

    Parameters
    ----------
    distance
        The distance of the code.
    basis
        The basis of the logical observable (or the type of parity check).
        It should be ``"X"`` or ``"Z"``.
    logical_qubit_label
        Label for the logical qubit, by default ``"L0"``.
    init_point
        Coordinates for the bottom left (i.e. southest west) data qubit.
        By default ``(1, 1)``.
    init_data_qubit_id
        Index for the bottom left (i.e. southest west) data qubit.
        By default ``1``, so the label is ``"D1"``.
    init_anc_qubit_id
        Index for the bottom left (i.e. southest west) ancilla qubit.
        By default ``1``, so the label is ``"A1"``.
    init_ind
        Minimum index that is going to be associated to a qubit.

    Returns
    -------
    Layout
        The layout of the code.
    """
    if not isinstance(init_point, tuple):
        raise TypeError(
            f"'init_point' must be a tuple, but {type(init_point)} was given."
        )
    if (len(init_point) != 2) or any(
        not isinstance(p, (float, int)) for p in init_point
    ):
        raise TypeError(f"'init_point' must have two elements that are floats or ints.")
    if not isinstance(logical_qubit_label, str):
        raise TypeError(
            "'logical_qubit_label' must be a string, "
            f"but {type(logical_qubit_label)} was given."
        )
    if not isinstance(init_data_qubit_id, int):
        raise TypeError(
            "'init_data_qubit_id' must be an int, "
            f"but {type(init_data_qubit_id)} was given."
        )
    if not isinstance(init_anc_qubit_id, int):
        raise TypeError(
            "'init_anc_qubit_id' must be an int, "
            f"but {type(init_anc_qubit_id)} was given."
        )
    if basis not in ["X", "Z"]:
        raise ValueError(f"'basis' must be 'Z' or 'X', but {basis} was given.")

    name = f"d={distance} repetition code layout."
    code = "repetition_code"
    description = None
    stab_type = f"{basis.lower()}_type"
    log_type = f"log_{basis.lower()}"
    other_basis = "X" if basis == "Z" else "Z"
    other_log_type = f"log_{other_basis.lower()}"

    int_order = {stab_type: ["south_east", "south_west"]}

    log_pauli = [f"D{i+init_data_qubit_id}" for i in range(distance)]

    layout_setup = {
        "name": name,
        "code": code,
        "logical_qubit_labels": [logical_qubit_label],
        "description": description,
        "distance": distance,
        "interaction_order": int_order,
        log_type: {logical_qubit_label: log_pauli},
        other_log_type: {logical_qubit_label: []},
    }

    layout_data = []
    neighbor_data = defaultdict(dict)
    ind = init_ind

    # change initial point because by default the code places the "D1" qubit
    # in the (1,1) point.
    init_point = (init_point[0] - 1, init_point[1] - 1)

    # data qubits
    for index in range(distance):
        qubit_info = dict(
            qubit=f"D{index}",
            role="data",
            coords=[0 + init_point[0], 2 * index + init_point[1]],
            stab_type=None,
            ind=ind,
        )
        layout_data.append(qubit_info)

        ind += 1

    # ancilla qubits
    for index in range(distance - 1):
        anc_qubit = f"A{index}"
        qubit_info = dict(
            qubit=anc_qubit,
            role="anc",
            coords=[1 + init_point[0], 2 * index + 1 + init_point[1]],
            stab_type=stab_type,
            ind=ind,
        )
        layout_data.append(qubit_info)

        neighbor_data[anc_qubit]["south_east"] = f"D{index}"
        neighbor_data[f"D{index}"]["north_west"] = anc_qubit

        neighbor_data[anc_qubit]["south_west"] = f"D{index+1}"
        neighbor_data[f"D{index+1}"]["north_east"] = anc_qubit

        ind += 1

    for qubit_info in layout_data:
        qubit = qubit_info["qubit"]
        qubit_info["neighbors"] = neighbor_data[qubit]

    layout_setup["layout"] = layout_data
    layout = Layout(layout_setup)
    return layout
