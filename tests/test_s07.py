"""test_e07.py — S07 Broadcasting test driver.

Provided by tinynum-starter. Do NOT modify.
The tester runs this file to verify your NDArray implementation.
"""

import os
import sys

# Ensure tinynum is importable regardless of how this script is invoked
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tinynum import NDArray


def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")


def fmt(v: float) -> str:
    return str(float(v))


def main() -> None:
    # --- broadcast_shapes ---
    emit("bc_same", str(list(NDArray.broadcast_shapes((2, 3), (2, 3)))))
    emit("bc_right_align", str(list(NDArray.broadcast_shapes((2, 3), (3,)))))
    emit("bc_both_expand", str(list(NDArray.broadcast_shapes((3, 1), (1, 4)))))
    emit("bc_3d", str(list(NDArray.broadcast_shapes((2, 1, 4), (3, 1)))))
    emit("bc_scalar_like", str(list(NDArray.broadcast_shapes((2, 3), (1,)))))

    try:
        NDArray.broadcast_shapes((2, 3), (4, 3))
        emit("bc_error", "NO_ERROR")
    except Exception:
        emit("bc_error", "ERROR")

    # --- broadcast_to ---
    row = NDArray.from_array([1, 2, 3], 1, 3)
    brow = row.broadcast_to(4, 3)
    emit("bt_shape", str(list(brow.get_shape())))
    emit("bt_get_0_1", fmt(brow.get(0, 1)))
    emit("bt_get_3_2", fmt(brow.get(3, 2)))
    emit("bt_toString", str(brow))

    # broadcast_to error: incompatible
    try:
        NDArray.from_array([1, 2, 3], 3).broadcast_to(4)
        emit("bt_error", "NO_ERROR")
    except Exception:
        emit("bt_error", "ERROR")

    # --- auto-broadcast add: matrix + row ---
    mat = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    bias = NDArray.from_array([10, 20, 30], 3)
    emit("add_mat_row", str(mat.add(bias)))

    # --- auto-broadcast mul: col * row (outer product) ---
    col = NDArray.from_array([1, 2, 3], 3, 1)
    row_vec = NDArray.from_array([10, 20, 30, 40], 1, 4)
    emit("mul_outer", str(col.mul(row_vec)))

    # --- auto-broadcast sub: 3D ---
    a3d = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 1, 3)
    b1d = NDArray.from_array([10, 20, 30], 3)
    emit("sub_3d", str(a3d.sub(b1d)))

    # --- auto-broadcast comparison: gt ---
    vals = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    thresh = NDArray.from_array([2, 3, 4], 1, 3)
    emit("gt_broadcast", str(vals.gt(thresh)))

    # --- broadcast_to is zero-copy (shares data) ---
    src = NDArray.from_array([7, 8, 9], 1, 3)
    view = src.broadcast_to(3, 3)
    # Mutate via set on the src (which shares data with view)
    src.set(99, 0, 0)
    emit("bt_zerocopy", fmt(view.get(0, 0)))

    # --- broadcast same shape is identity ---
    same = NDArray.from_array([1, 2], 2)
    emit("bt_identity", str(same.broadcast_to(2)))

    # --- scalar-like broadcast [1] to [4] ---
    scalar = NDArray.from_array([5], 1)
    expanded = scalar.broadcast_to(4)
    emit("bt_scalar", str(expanded))


if __name__ == "__main__":
    main()
