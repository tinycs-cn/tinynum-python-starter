import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray, DType

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # ---- tril default diagonal ----
    t0 = NDArray.tril(3, 0)
    emit("tril_default", str(t0))

    # ---- tril with diagonal=1 ----
    t1 = NDArray.tril(3, 1)
    emit("tril_diag1", str(t1))

    # ---- triu default diagonal ----
    u0 = NDArray.triu(3, 0)
    emit("triu_default", str(u0))

    # ---- triu with diagonal=-1 ----
    u1 = NDArray.triu(3, -1)
    emit("triu_diag_neg1", str(u1))

    # ---- norm axis=1 (row-wise) ----
    n1 = NDArray.from_array([3,4,6,8], 2, 2)
    emit("norm_axis1", str(n1.norm(1)))

    # ---- norm axis=0 (col-wise) ----
    n1b = NDArray.from_array([3,5,4,12], 2, 2)
    emit("norm_axis0", str(n1b.norm(0)))

    # ---- diff axis=0 on 1D ----
    d1 = NDArray.from_array([1,3,6,10], 4)
    emit("diff_1d", str(d1.diff(0)))

    # ---- diff axis=1 on 2D ----
    d2 = NDArray.from_array([1,3,6,10,2,5,9,14], 2, 4)
    emit("diff_axis1", str(d2.diff(1)))
    emit("diff_axis1_shape", str(list(d2.diff(1).get_shape())))

    # ---- diff axis=0 on 2D ----
    emit("diff_axis0", str(d2.diff(0)))

    # ---- percentile 50 ----
    p1 = NDArray.from_array([1,2,3,4,5], 5)
    emit("percentile_50", str(p1.percentile(50)))

    # ---- percentile 0 and 100 ----
    emit("percentile_0", str(p1.percentile(0)))
    emit("percentile_100", str(p1.percentile(100)))

    # ---- percentile 25 ----
    emit("percentile_25", str(p1.percentile(25)))

    # ---- argsort 1D ----
    a1 = NDArray.from_array([30, 10, 20], 3)
    emit("argsort_1d", str(a1.argsort(0)))

    # ---- argsort 2D axis=1 ----
    a2 = NDArray.from_array([30,10,20,5,15,1], 2, 3)
    emit("argsort_2d_axis1", str(a2.argsort(1)))

    # ---- unique ----
    uq = NDArray.from_array([3,1,2,1,3,2], 6)
    emit("unique", str(uq.unique()))
    emit("unique_shape", str(list(uq.unique().get_shape())))

    # ---- allClose true ----
    ac1 = NDArray.from_array([1.0, 2.0, 3.0], 3)
    ac2 = NDArray.from_array([1.0001, 2.0001, 3.0001], 3)
    emit("allclose_true", str(ac1.all_close(ac2, 0.001)).lower())

    # ---- allClose false ----
    ac3 = NDArray.from_array([1.0, 2.0, 3.0], 3)
    ac4 = NDArray.from_array([1.0, 2.5, 3.0], 3)
    emit("allclose_false", str(ac3.all_close(ac4, 0.001)).lower())

    # ---- allClose shape mismatch ----
    ac5 = NDArray.from_array([1.0, 2.0], 2)
    emit("allclose_shape_diff", str(ac1.all_close(ac5, 0.001)).lower())

    # ---- astype float32 -> int8 ----
    at1 = NDArray.from_array([150.5, -200.3, 50.0, 0.7], 4)
    emit("astype_int8", str(at1.astype(DType.INT8)))

    # ---- astype int8 -> float32 (identity on already-int-valued) ----
    at2 = NDArray.from_array([127, -128, 50], 3)
    emit("astype_float32", str(at2.astype(DType.FLOAT32)))

    # ---- softmax integration test ----
    x = NDArray.from_array([1,2,3,1,2,3], 2, 3)
    max_val = x.max(1, keep_dims=True)
    shifted = x.sub(max_val)
    exp_val = shifted.exp()
    sum_exp = exp_val.sum(1, keep_dims=True)
    softmax = exp_val.div(sum_exp)

    # Check row sums ≈ 1.0
    row_sums = softmax.sum(1, keep_dims=False)
    expected_ones = NDArray.from_array([1.0, 1.0], 2)
    emit("softmax_row_sum", str(row_sums.all_close(expected_ones, 0.0001)).lower())

    # Check softmax values are reasonable (max element has largest prob)
    s0 = softmax.get(0, 0)
    s1 = softmax.get(0, 1)
    s2 = softmax.get(0, 2)
    emit("softmax_order", str(s2 > s1 and s1 > s0).lower())

if __name__ == "__main__":
    main()
