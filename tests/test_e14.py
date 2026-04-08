import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # ---- indexSelect axis=0 (row select) ----
    w = NDArray.from_array([1,2,3,4,5,6,7,8,9,10,11,12], 4, 3)
    r0 = w.index_select(0, [2, 0, 3])
    emit("index_select_axis0", str(r0))
    emit("index_select_axis0_shape", str(list(r0.get_shape())))

    # ---- indexSelect axis=1 (column select) ----
    r1 = w.index_select(1, [2, 0])
    emit("index_select_axis1", str(r1))
    emit("index_select_axis1_shape", str(list(r1.get_shape())))

    # ---- indexSelect 1D ----
    v = NDArray.from_array([10, 20, 30, 40, 50], 5)
    r2 = v.index_select(0, [4, 1, 1, 3])
    emit("index_select_1d", str(r2))

    # ---- indexSelect duplicate indices ----
    r3 = w.index_select(0, [0, 0, 2])
    emit("index_select_dup", str(r3))

    # ---- scatterAdd basic ----
    grad = NDArray.from_array([0,0,0,0,0,0,0,0,0,0,0,0], 4, 3)
    src = NDArray.from_array([1,1,1,2,2,2], 2, 3)
    grad.scatter_add(0, [2, 0], src)
    emit("scatter_add_basic", str(grad))

    # ---- scatterAdd duplicate indices (accumulation) ----
    grad2 = NDArray.from_array([0,0,0,0,0,0], 2, 3)
    src2 = NDArray.from_array([1,2,3,4,5,6,7,8,9], 3, 3)
    grad2.scatter_add(0, [0, 1, 0], src2)
    emit("scatter_add_dup", str(grad2))

    # ---- maskedFill 2D ----
    scores = NDArray.from_array([1,2,3,4,5,6], 2, 3)
    mask = NDArray.from_array([0,0,1,1,0,0], 2, 3)
    filled = scores.masked_fill(mask, -999.0)
    emit("masked_fill_2d", str(filled))

    # ---- maskedFill 1D ----
    v2 = NDArray.from_array([10, 20, 30], 3)
    m2 = NDArray.from_array([1, 0, 1], 3)
    emit("masked_fill_1d", str(v2.masked_fill(m2, 0)))

    # ---- maskedFill no change (all zeros mask) ----
    m3 = NDArray.from_array([0, 0, 0], 3)
    emit("masked_fill_none", str(v2.masked_fill(m3, -99)))

    # ---- where 2D ----
    cond = NDArray.from_array([1,0,0,1], 2, 2)
    x = NDArray.from_array([10,20,30,40], 2, 2)
    y = NDArray.from_array([-1,-2,-3,-4], 2, 2)
    emit("where_2d", str(NDArray.where(cond, x, y)))

    # ---- where 1D ----
    c1 = NDArray.from_array([0, 1, 0, 1, 1], 5)
    x1 = NDArray.from_array([1, 2, 3, 4, 5], 5)
    y1 = NDArray.from_array([-1, -2, -3, -4, -5], 5)
    emit("where_1d", str(NDArray.where(c1, x1, y1)))

    # ---- error: maskedFill shape mismatch ----
    try:
        a = NDArray.from_array([1,2,3,4], 2, 2)
        bad_mask = NDArray.from_array([1,2,3], 3)
        a.masked_fill(bad_mask, 0)
        emit("masked_fill_shape_error", "NO_ERROR")
    except Exception:
        emit("masked_fill_shape_error", "ERROR")

    # ---- error: where shape mismatch ----
    try:
        a = NDArray.from_array([1,0], 2)
        b = NDArray.from_array([1,2,3], 3)
        c = NDArray.from_array([4,5,6], 3)
        NDArray.where(a, b, c)
        emit("where_shape_error", "NO_ERROR")
    except Exception:
        emit("where_shape_error", "ERROR")

if __name__ == "__main__":
    main()
