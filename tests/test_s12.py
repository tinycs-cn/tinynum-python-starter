import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # ---- arange ----
    a1 = NDArray.arange(0, 5, 1)
    emit("arange_basic", str(a1))
    emit("arange_basic_shape", str(list(a1.get_shape())))

    a2 = NDArray.arange(1, 2, 0.3)
    emit("arange_float_len", str(a2.get_shape()[0]))

    # ---- linspace ----
    l1 = NDArray.linspace(0, 1, 5)
    emit("linspace_basic", str(l1))
    emit("linspace_basic_shape", str(list(l1.get_shape())))

    l2 = NDArray.linspace(0, 10, 3)
    emit("linspace_three", str(l2))

    l3 = NDArray.linspace(5, 5, 1)
    emit("linspace_single", str(l3))

    # ---- eye ----
    e1 = NDArray.eye(3)
    emit("eye_3", str(e1))
    emit("eye_3_shape", str(list(e1.get_shape())))

    e2 = NDArray.eye(1)
    emit("eye_1", str(e2))

    # ---- diag ----
    v = NDArray.from_array([3, 5, 7], 3)
    d1 = NDArray.diag(v)
    emit("diag_basic", str(d1))
    emit("diag_basic_shape", str(list(d1.get_shape())))

    # ---- randn ----
    rn = NDArray.randn(2, 3)
    emit("randn_shape", str(list(rn.get_shape())))

    rn_large = NDArray.randn(10000)
    rn_mean = rn_large.mean_all()
    emit("randn_mean_near_zero", "true" if abs(rn_mean) < 0.1 else "false")

    # ---- rand ----
    rd = NDArray.rand(3, 4)
    emit("rand_shape", str(list(rd.get_shape())))

    rd_flat = NDArray.rand(1000)
    rand_in_range = all(0 <= rd_flat.get(i) < 1 for i in range(1000))
    emit("rand_in_range", str(rand_in_range).lower())

    # ---- uniform ----
    u = NDArray.uniform(-2, 3, 2, 5)
    emit("uniform_shape", str(list(u.get_shape())))

    u_flat = NDArray.uniform(-2, 3, 1000)
    uniform_in_range = all(-2 <= u_flat.get(i) < 3 for i in range(1000))
    emit("uniform_in_range", str(uniform_in_range).lower())

    # ---- shuffle ----
    idx = [0, 1, 2, 3, 4]
    NDArray.shuffle(idx)
    emit("shuffle_length", str(len(idx)))
    emit("shuffle_elements_preserved", str(sorted(idx) == [0, 1, 2, 3, 4]).lower())

    # ---- fill ----
    f = NDArray.from_array([1, 2, 3, 4, 5, 6], 2, 3)
    f.fill(0)
    emit("fill_zeros", str(f))

    f2 = NDArray.from_array([1, 2, 3, 4], 2, 2)
    f2.fill(7)
    emit("fill_value", str(f2))

    # ---- eye identity property: eye(3) * x == x ----
    x = NDArray.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3)
    identity = NDArray.eye(3)
    product = identity.matmul(x)
    emit("eye_identity_property", str(product))

    # ---- error: diag with non-1D ----
    try:
        NDArray.diag(NDArray.from_array([1, 2, 3, 4], 2, 2))
        emit("diag_error_non1d", "NO_ERROR")
    except Exception:
        emit("diag_error_non1d", "ERROR")

if __name__ == "__main__":
    main()
