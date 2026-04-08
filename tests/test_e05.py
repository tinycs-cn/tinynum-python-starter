"""test_e05.py — E05 Unary Math test driver.

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
    # --- neg ---
    a = NDArray.from_array([1, -2, 3], 3)
    emit("neg_toString", str(a.neg()))

    # --- abs ---
    emit("abs_toString", str(a.abs()))

    # --- exp of zeros ---
    z = NDArray.from_array([0, 0, 0, 0, 0, 0], 2, 3)
    emit("exp_zeros", str(z.exp()))

    # --- log of ones ---
    ones = NDArray.from_array([1, 1, 1, 1], 2, 2)
    emit("log_ones", str(ones.log()))

    # --- sqrt ---
    sq = NDArray.from_array([0, 1, 4, 9], 4)
    emit("sqrt_toString", str(sq.sqrt()))

    # --- square ---
    b = NDArray.from_array([-2, 0, 3], 3)
    emit("square_toString", str(b.square()))

    # --- tanh(0) ---
    t = NDArray.from_array([0], 1)
    emit("tanh_zero", fmt(t.tanh().get(0)))

    # --- sin(0) and cos(0) ---
    emit("sin_zero", fmt(t.sin().get(0)))
    emit("cos_zero", fmt(t.cos().get(0)))

    # --- sign ---
    s = NDArray.from_array([-5, 0, 7], 3)
    emit("sign_toString", str(s.sign()))

    # --- round ---
    r = NDArray.from_array([1.4, 1.6, -0.5, 2.3], 4)
    emit("round_toString", str(r.round()))

    # --- clip ---
    c = NDArray.from_array([-3, -1, 0, 1, 3], 5)
    emit("clip_toString", str(c.clip(-2, 2)))

    # --- pow ---
    p = NDArray.from_array([1, 4, 9], 3)
    emit("pow_half", str(p.pow(0.5)))

    # --- unary preserves shape (returns new array) ---
    orig = NDArray.from_array([1, 2, 3], 3)
    negated = orig.neg()
    emit("unary_independent", fmt(orig.get(0)))

    # --- unary on transposed array ---
    m = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    emit("unary_transposed", str(m.transpose().square()))


if __name__ == "__main__":
    main()
