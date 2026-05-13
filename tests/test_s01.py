"""test_e01.py — S01 Storage & Shape test driver.

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


def main() -> None:
    # --- zeros ---
    z = NDArray.zeros(2, 3)
    emit("zeros_size", str(z.size()))
    emit("zeros_ndim", str(z.ndim()))
    emit("zeros_shape", shape_str(z.get_shape()))
    emit("zeros_toString", str(z))

    # --- ones ---
    o = NDArray.ones(3, 4)
    emit("ones_size", str(o.size()))
    emit("ones_toString", str(NDArray.ones(2, 3)))

    # --- from_array 2D ---
    a = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("fromArray_2d_toString", str(a))

    # --- full ---
    f = NDArray.full(7.0, 2, 2)
    emit("full_toString", str(f))

    # --- zeros_like / ones_like ---
    base = NDArray.from_array([1, 2, 3, 4], 2, 2)
    emit("zerosLike_toString", str(NDArray.zeros_like(base)))
    emit("onesLike_toString", str(NDArray.ones_like(base)))

    # --- 1D ---
    v = NDArray.from_array([1, 2, 3], 3)
    emit("1d_toString", str(v))

    # --- 3D ---
    t = NDArray.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 2, 3, 2)
    emit("3d_ndim", str(t.ndim()))
    emit("3d_size", str(t.size()))
    emit("3d_toString", str(t))

    # --- error: data/shape mismatch ---
    try:
        NDArray.from_array([1, 2, 3], 2, 2)
        emit("error_mismatch", "NO_ERROR")
    except (ValueError, Exception):
        emit("error_mismatch", "ERROR")

    # --- 数据隔离：修改原始列表不影响 NDArray ---
    raw_data = [1.0, 2.0, 3.0, 4.0]
    isolated = NDArray.from_array(raw_data, 2, 2)
    raw_data[0] = 99.0
    emit("data_isolation", str(isolated))

    # --- get_shape() 返回副本：tuple 本身不可变，修改无影响 ---
    sq = NDArray.zeros(2, 3)
    emit("shape_copy", "OK" if sq.get_shape()[0] == 2 else "FAIL")


if __name__ == "__main__":
    main()
