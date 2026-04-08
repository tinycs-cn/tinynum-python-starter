import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # Test data: [[1,2,3],[4,5,6]] shape [2,3]
    a = NDArray.from_array([1,2,3,4,5,6], 2, 3)

    # Global reduction
    emit("sum_global", str(float(a.sum_all())))
    emit("mean_global", str(float(a.mean_all())))

    # sum along axis (no keepDims)
    s0 = a.sum(0, False)
    emit("sum_axis0_nokeep", str(s0))
    emit("sum_axis0_shape_nokeep", str(list(s0.get_shape())))

    s1 = a.sum(1, False)
    emit("sum_axis1_nokeep", str(s1))
    emit("sum_axis1_shape_nokeep", str(list(s1.get_shape())))

    # sum along axis (keepDims=true)
    s0k = a.sum(0, True)
    emit("sum_axis0_keep", str(s0k))
    emit("sum_axis0_shape_keep", str(list(s0k.get_shape())))

    s1k = a.sum(1, True)
    emit("sum_axis1_keep", str(s1k))
    emit("sum_axis1_shape_keep", str(list(s1k.get_shape())))

    # mean along axis
    m1 = a.mean(1, False)
    emit("mean_axis1_nokeep", str(m1))

    m0k = a.mean(0, True)
    emit("mean_axis0_keep", str(m0k))

    # Negative axis
    s_neg = a.sum(-1, False)
    emit("sum_neg_axis", str(s_neg))

    # 3D array: [[[1,2],[3,4]],[[5,6],[7,8]]] shape [2,2,2]
    b = NDArray.from_array([1,2,3,4,5,6,7,8], 2, 2, 2)
    b1 = b.sum(1, False)
    emit("sum_3d_axis1", str(b1))
    emit("sum_3d_axis1_shape", str(list(b1.get_shape())))

    # Multi-axis sum: axes=[0,2] on shape [2,2,2]
    bm = b.sum_axes([0, 2], False)
    emit("sum_multi_axes", str(bm))
    emit("sum_multi_axes_shape", str(list(bm.get_shape())))

    # 1D array sum
    c = NDArray.from_array([10, 20, 30], 3)
    c0 = c.sum(0, False)
    emit("sum_1d", str(float(c0.sum_all())))

    # Error: invalid axis
    try:
        a.sum(3, False)
        emit("sum_axis_error", "NO_ERROR")
    except Exception:
        emit("sum_axis_error", "ERROR")

if __name__ == "__main__":
    main()
