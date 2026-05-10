"""test_e06.py — S06 Binary Ops & Comparisons test driver.

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
    a = NDArray.from_array([2, 4, 6, 8, 10, 12], 2, 3)
    b = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)

    # --- Arithmetic (NDArray) ---
    emit("add_toString", str(a.add(b)))
    emit("sub_toString", str(a.sub(b)))
    emit("mul_toString", str(a.mul(b)))
    emit("div_toString", str(a.div(b)))

    p1 = NDArray.from_array([2, 3, 4], 3)
    p2 = NDArray.from_array([3, 2, 0], 3)
    emit("pow_toString", str(p1.pow_array(p2)))

    m1 = NDArray.from_array([1, 5, 3], 3)
    m2 = NDArray.from_array([4, 2, 3], 3)
    emit("maximum_toString", str(m1.maximum(m2)))

    # --- Arithmetic (scalar) ---
    c = NDArray.from_array([10, 20, 30], 3)
    emit("add_scalar", str(c.add_scalar(5)))
    emit("sub_scalar", str(c.sub_scalar(5)))
    emit("mul_scalar", str(c.mul_scalar(2)))
    emit("div_scalar", str(c.div_scalar(10)))

    # --- Comparisons (NDArray) ---
    x = NDArray.from_array([1, 2, 3, 4], 4)
    y = NDArray.from_array([1, 3, 2, 4], 4)
    emit("eq_toString", str(x.eq(y)))
    emit("neq_toString", str(x.neq(y)))
    emit("gt_toString", str(x.gt(y)))
    emit("gte_toString", str(x.gte(y)))
    emit("lt_toString", str(x.lt(y)))
    emit("lte_toString", str(x.lte(y)))

    # --- Comparisons (scalar) ---
    s = NDArray.from_array([1, 2, 3, 4, 5], 5)
    emit("eq_scalar", str(s.eq_scalar(3)))
    emit("neq_scalar", str(s.neq_scalar(3)))
    emit("gt_scalar", str(s.gt_scalar(3)))
    emit("gte_scalar", str(s.gte_scalar(3)))
    emit("lt_scalar", str(s.lt_scalar(3)))
    emit("lte_scalar", str(s.lte_scalar(3)))

    # --- Binary returns new array (originals unchanged) ---
    orig1 = NDArray.from_array([1, 2, 3], 3)
    orig2 = NDArray.from_array([4, 5, 6], 3)
    sum_arr = orig1.add(orig2)
    emit("binary_independent", fmt(orig1.get(0)))

    # --- Binary on transposed arrays ---
    t1 = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    t2 = NDArray.from_array([10, 20, 30, 40, 50, 60], 3, 2)
    emit("binary_transposed", str(t1.transpose().add(t2)))

    # --- Shape mismatch ---
    try:
        NDArray.from_array([1, 2, 3], 3).add(NDArray.from_array([1, 2], 2))
        emit("shape_mismatch", "NO_ERROR")
    except Exception:
        emit("shape_mismatch", "ERROR")


if __name__ == "__main__":
    main()
