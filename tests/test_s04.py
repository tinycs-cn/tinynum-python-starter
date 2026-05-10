"""test_e04.py — S04 Transpose test driver.

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


def shape_str(shape: tuple[int, ...]) -> str:
    return str(list(shape))


def fmt(v: float) -> str:
    return str(float(v))


def main() -> None:
    # --- 2D transpose shape ---
    a = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    t = a.transpose()
    emit("transpose_2d_shape", shape_str(t.get_shape()))

    # --- 2D transpose toString ---
    emit("transpose_2d_toString", str(t))

    # --- get(i,j) == transpose().get(j,i) ---
    emit("transpose_get_equiv", fmt(t.get(1, 0)))

    # --- 2D transpose zero-copy ---
    a2 = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    t2 = a2.transpose()
    t2.set(99.0, 0, 1)
    emit("transpose_zerocopy", fmt(a2.get(1, 0)))

    # --- transposed is not contiguous ---
    a3 = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("transpose_not_contiguous", str(a3.transpose().is_contiguous()).lower())

    # --- N-D transpose(axes) shape ---
    b = NDArray.from_array([0] * 24, 2, 3, 4)
    bt = b.transpose(2, 0, 1)
    emit("transpose_nd_shape", shape_str(bt.get_shape()))

    # --- N-D transpose identity ---
    c = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    ci = c.transpose(0, 1)
    emit("transpose_identity_toString", str(ci))

    # --- swapAxes 3D ---
    d = NDArray.from_array([0] * 24, 2, 3, 4)
    ds = d.swap_axes(0, 2)
    emit("swapAxes_shape", shape_str(ds.get_shape()))

    # --- swapAxes same axis is identity ---
    e = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    es = e.swap_axes(0, 0)
    emit("swapAxes_same_toString", str(es))

    # --- error: transpose() on non-2D ---
    try:
        f = NDArray.from_array([0] * 24, 2, 3, 4)
        f.transpose()
        emit("error_transpose_non2d", "NO_ERROR")
    except Exception:
        emit("error_transpose_non2d", "ERROR")

    # --- error: invalid axes ---
    try:
        g = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
        g.transpose(0, 0)
        emit("error_invalid_axes", "NO_ERROR")
    except Exception:
        emit("error_invalid_axes", "ERROR")


if __name__ == "__main__":
    main()
