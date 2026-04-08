"""test_e03.py — E03 Reshape test driver.

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
    return ",".join(str(s) for s in shape)


def fmt(v: float) -> str:
    return str(float(v))


def main() -> None:
    a = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)

    # --- reshape basic ---
    b = a.reshape(3, 2)
    emit("reshape_shape", shape_str(b.get_shape()))
    emit("reshape_toString", str(b))

    # --- reshape -1 ---
    c = a.reshape(3, -1)
    emit("reshape_neg1_shape", shape_str(c.get_shape()))

    # --- reshape -1 in 3D ---
    d = NDArray.from_array(list(range(1, 13)), 2, 6)
    e = d.reshape(2, -1, 3)
    emit("reshape_neg1_3d_shape", shape_str(e.get_shape()))

    # --- reshape zero-copy ---
    f = a.reshape(6)
    f.set(99.0, 0)
    emit("reshape_zerocopy", fmt(a.get(0, 0)))

    # --- flatten ---
    g = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    flat = g.flatten()
    emit("flatten_shape", shape_str(flat.get_shape()))
    emit("flatten_toString", str(flat))

    # --- duplicate ---
    h = NDArray.from_array([1, 2, 3, 4], 2, 2)
    dup = h.duplicate()
    emit("duplicate_toString", str(dup))
    dup.set(99.0, 0, 0)
    emit("duplicate_independent", fmt(h.get(0, 0)))

    # --- error: reshape size mismatch ---
    try:
        a.reshape(4, 4)
        emit("error_reshape_size", "NO_EXCEPTION")
    except (ValueError, Exception):
        emit("error_reshape_size", "EXCEPTION")


if __name__ == "__main__":
    main()
