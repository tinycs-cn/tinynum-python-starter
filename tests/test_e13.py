import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # ---- concatenate axis=0 ----
    a = NDArray.from_array([1,2,3,4], 2, 2)
    b = NDArray.from_array([5,6,7,8], 2, 2)
    c0 = NDArray.concatenate([a, b], 0)
    emit("concat_axis0", str(c0))
    emit("concat_axis0_shape", str(list(c0.get_shape())))

    # ---- concatenate axis=1 ----
    c1 = NDArray.concatenate([a, b], 1)
    emit("concat_axis1", str(c1))
    emit("concat_axis1_shape", str(list(c1.get_shape())))

    # ---- concatenate different sizes on concat axis ----
    d = NDArray.from_array([9,10], 1, 2)
    c2 = NDArray.concatenate([a, d], 0)
    emit("concat_diff_size", str(c2))
    emit("concat_diff_size_shape", str(list(c2.get_shape())))

    # ---- concatenate 1D ----
    e = NDArray.from_array([1,2,3], 3)
    f = NDArray.from_array([4,5], 2)
    c3 = NDArray.concatenate([e, f], 0)
    emit("concat_1d", str(c3))

    # ---- stack axis=0 ----
    g = NDArray.from_array([1,2,3], 3)
    h = NDArray.from_array([4,5,6], 3)
    s0 = NDArray.stack([g, h], 0)
    emit("stack_axis0", str(s0))
    emit("stack_axis0_shape", str(list(s0.get_shape())))

    # ---- stack axis=1 ----
    s1 = NDArray.stack([g, h], 1)
    emit("stack_axis1", str(s1))
    emit("stack_axis1_shape", str(list(s1.get_shape())))

    # ---- stack 2D ----
    i = NDArray.from_array([1,2,3,4], 2, 2)
    j = NDArray.from_array([5,6,7,8], 2, 2)
    s2 = NDArray.stack([i, j], 0)
    emit("stack_2d_shape", str(list(s2.get_shape())))

    # ---- pad ----
    p = NDArray.from_array([1,2,3,4,5,6], 2, 3)
    p1 = p.pad([(1,1),(0,0)], 0)
    emit("pad_rows", str(p1))
    emit("pad_rows_shape", str(list(p1.get_shape())))

    # ---- pad with value ----
    p2 = p.pad([(0,0),(1,2)], -1)
    emit("pad_cols_value", str(p2))
    emit("pad_cols_shape", str(list(p2.get_shape())))

    # ---- pad 1D ----
    q = NDArray.from_array([1,2,3], 3)
    q1 = q.pad([(2,1)], 0)
    emit("pad_1d", str(q1))

    # ---- flip axis=0 ----
    r = NDArray.from_array([1,2,3,4,5,6], 2, 3)
    emit("flip_axis0", str(r.flip(0)))

    # ---- flip axis=1 ----
    emit("flip_axis1", str(r.flip(1)))

    # ---- flip 1D ----
    t = NDArray.from_array([1,2,3,4,5], 5)
    emit("flip_1d", str(t.flip(0)))

    # ---- error: concat shape mismatch ----
    try:
        x1 = NDArray.from_array([1,2,3,4], 2, 2)
        x2 = NDArray.from_array([1,2,3,4,5,6], 2, 3)
        NDArray.concatenate([x1, x2], 0)
        emit("concat_shape_error", "NO_ERROR")
    except Exception:
        emit("concat_shape_error", "ERROR")

    # ---- error: pad wrong padWidth length ----
    try:
        y = NDArray.from_array([1,2,3,4], 2, 2)
        y.pad([(1,1)], 0)  # 1 pad for 2D
        emit("pad_dim_error", "NO_ERROR")
    except Exception:
        emit("pad_dim_error", "ERROR")

if __name__ == "__main__":
    main()
