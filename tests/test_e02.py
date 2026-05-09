"""test_e02.py — E02 Strides & Indexing test driver.

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


def strides_str(strides: tuple[int, ...]) -> str:
    return ",".join(str(s) for s in strides)


def fmt(v: float) -> str:
    return str(float(v))


def main() -> None:
    # --- compute_strides ---
    emit("strides_2d", strides_str(NDArray.compute_strides((2, 3))))
    emit("strides_3d", strides_str(NDArray.compute_strides((3, 4, 5))))
    emit("strides_1d", strides_str(NDArray.compute_strides((5,))))

    # --- get: 2D ---
    a = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("get_2d_00", fmt(a.get(0, 0)))
    emit("get_2d_02", fmt(a.get(0, 2)))
    emit("get_2d_10", fmt(a.get(1, 0)))
    emit("get_2d_12", fmt(a.get(1, 2)))

    # --- get: 3D ---
    b = NDArray.from_array(list(range(1, 25)), 2, 3, 4)
    emit("get_3d_000", fmt(b.get(0, 0, 0)))
    emit("get_3d_123", fmt(b.get(1, 2, 3)))
    emit("get_3d_012", fmt(b.get(0, 1, 2)))

    # --- set ---
    a.set(99.0, 1, 1)
    emit("set_get", fmt(a.get(1, 1)))
    emit("set_toString", str(a))

    # --- is_contiguous ---
    c = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("isContiguous_fresh", str(c.is_contiguous()).lower())

    # --- error: wrong index count ---
    try:
        a.get(1)
        emit("error_wrong_indices", "NO_ERROR")
    except (ValueError, IndexError, Exception):
        emit("error_wrong_indices", "ERROR")

    # --- error: out-of-bounds index ---
    try:
        a.get(10, 0)
        emit("error_out_of_bounds", "NO_ERROR")
    except (ValueError, IndexError, Exception):
        emit("error_out_of_bounds", "ERROR")


if __name__ == "__main__":
    main()
