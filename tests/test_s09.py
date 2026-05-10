import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # Test data: [[3,1,4],[1,5,9]] shape [2,3]
    a = NDArray.from_array([3,1,4,1,5,9], 2, 3)

    # max
    emit("max_axis1", str(a.max(1, False)))
    max_keep = a.max(0, True)
    emit("max_axis0_keep", str(max_keep))
    emit("max_axis0_keep_shape", str(list(max_keep.get_shape())))

    # min
    emit("min_axis0", str(a.min(0, False)))
    emit("min_axis1_keep", str(a.min(1, True)))

    # argmax
    emit("argmax_axis1", str(a.argmax(1)))
    emit("argmax_axis0", str(a.argmax(0)))

    # argmin
    emit("argmin_axis1", str(a.argmin(1)))
    emit("argmin_axis0", str(a.argmin(0)))

    # prod — use [[1,2,3],[4,5,6]] shape [2,3]
    b = NDArray.from_array([1,2,3,4,5,6], 2, 3)
    emit("prod_axis1", str(b.prod(1)))
    emit("prod_axis0", str(b.prod(0)))

    # var / std — use [[1,3],[2,4]] shape [2,2] (exact results)
    c = NDArray.from_array([1,3,2,4], 2, 2)
    emit("var_axis1", str(c.var(1, False)))
    emit("var_axis0_keep", str(c.var(0, True)))
    emit("std_axis1", str(c.std(1, False)))
    emit("std_axis0", str(c.std(0, False)))

    # countNonZero
    d = NDArray.from_array([0,1,0,3,5], 5)
    emit("count_nonzero", str(d.count_nonzero()))

    # countNonZero 2D
    e = NDArray.from_array([0,1,2,0,3,4], 3, 2)
    emit("count_nonzero_2d", str(e.count_nonzero()))

    # Negative axis
    emit("max_neg_axis", str(a.max(-1, False)))

    # 3D: [[[3,1],[4,1]],[[5,9],[2,6]]] shape [2,2,2]
    f = NDArray.from_array([3,1,4,1,5,9,2,6], 2, 2, 2)
    emit("max_3d", str(f.max(1, False)))

    # Error: invalid axis
    try:
        a.max(3, False)
        emit("max_axis_error", "NO_ERROR")
    except Exception:
        emit("max_axis_error", "ERROR")

if __name__ == "__main__":
    main()
