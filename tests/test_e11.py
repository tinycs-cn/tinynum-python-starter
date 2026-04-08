import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray
from tinynum.slice import Slice

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # ---- slice 2D ----
    a = NDArray.from_array([1,2,3,4,5,6,7,8,9], 3, 3)
    s1 = a.slice(Slice.of(0, 2), Slice.of(1, 3))
    emit("slice_2d", str(s1))
    emit("slice_2d_shape", str(list(s1.get_shape())))

    # ---- slice with step ----
    b = NDArray.from_array([0,1,2,3,4,5], 6)
    s2 = b.slice(Slice.of(0, 6, 2))
    emit("slice_step", str(s2))
    emit("slice_step_shape", str(list(s2.get_shape())))

    # ---- Slice.all() ----
    s3 = a.slice(Slice.of(1, 2), Slice.all())
    emit("slice_all", str(s3))

    # ---- view shared memory ----
    orig = NDArray.from_array([1,2,3,4], 2, 2)
    view = orig.slice(Slice.of(0, 1), Slice.all())
    view.set(99.0, 0, 0)
    emit("slice_view_shared", str(float(orig.get(0, 0))))

    # ---- 3D slice ----
    # [[[1,2],[3,4]],[[5,6],[7,8]]] shape [2,2,2]
    c = NDArray.from_array([1,2,3,4,5,6,7,8], 2, 2, 2)
    s4 = c.slice(Slice.all(), Slice.of(0, 1), Slice.all())
    emit("slice_3d", str(s4))
    emit("slice_3d_shape", str(list(s4.get_shape())))

    # ---- expandDims ----
    d = NDArray.from_array([1,2,3,4,5,6], 2, 3)
    emit("expand_dims_0", str(list(d.expand_dims(0).get_shape())))
    emit("expand_dims_1", str(list(d.expand_dims(1).get_shape())))
    emit("expand_dims_last", str(list(d.expand_dims(2).get_shape())))

    # expandDims is a view
    e0 = NDArray.from_array([10,20], 2)
    e1 = e0.expand_dims(0)
    e1.set(99.0, 0, 0)
    emit("expand_dims_view", str(float(e0.get(0))))

    # ---- squeeze ----
    f = NDArray.from_array([1,2,3], 1, 3, 1)
    sq1 = f.squeeze_axis(0)
    emit("squeeze_axis", str(sq1))
    emit("squeeze_axis_shape", str(list(sq1.get_shape())))

    sq2 = f.squeeze()
    emit("squeeze_all", str(sq2))
    emit("squeeze_all_shape", str(list(sq2.get_shape())))

    # ---- errors ----
    try:
        f.squeeze_axis(1)  # axis 1 has size 3
        emit("squeeze_error", "NO_ERROR")
    except Exception:
        emit("squeeze_error", "ERROR")

    try:
        a.slice(Slice.of(0, 2))  # 1 slice for 2D array
        emit("slice_range_error", "NO_ERROR")
    except Exception:
        emit("slice_range_error", "ERROR")

if __name__ == "__main__":
    main()
