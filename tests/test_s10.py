import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinynum import NDArray

def emit(test_name: str, result: str) -> None:
    print(f"TEST:{test_name}")
    print(f"RESULT:{result}")

def main() -> None:
    # ---- dot ----
    a = NDArray.from_array([1,2,3], 3)
    b = NDArray.from_array([4,5,6], 3)
    dot_result = a.dot(b)
    emit("dot_1d", str(float(dot_result.get())))
    emit("dot_shape", str(list(dot_result.get_shape())))

    # ---- matmul 2D ----
    A = NDArray.from_array([1,2,3,4], 2, 2)
    B = NDArray.from_array([5,6,7,8], 2, 2)
    emit("matmul_2d", str(A.matmul(B)))
    emit("matmul_2d_shape", str(list(A.matmul(B).get_shape())))

    # ---- matmul rectangular ----
    C = NDArray.from_array([1,2,3,4,5,6], 2, 3)
    D = NDArray.from_array([1,2,3,4,5,6], 3, 2)
    rect_result = C.matmul(D)
    emit("matmul_rect", str(rect_result))
    emit("matmul_rect_shape", str(list(rect_result.get_shape())))

    # ---- matmul identity ----
    eye = NDArray.from_array([1,0,0,0,1,0,0,0,1], 3, 3)
    X = NDArray.from_array([2,3,4,5,6,7,8,9,10], 3, 3)
    emit("matmul_identity", str(eye.matmul(X)))

    # ---- batch matmul ----
    # [2,2,3] @ [2,3,2]
    bA = NDArray.from_array([1,2,3,4,5,6,7,8,9,10,11,12], 2, 2, 3)
    bB = NDArray.from_array([1,0,0,1,1,0,0,1,1,0,0,1], 2, 3, 2)
    batch_result = bA.matmul(bB)
    emit("matmul_batch", str(batch_result))
    emit("matmul_batch_shape", str(list(batch_result.get_shape())))

    # ---- batch broadcast: [2,2,3] @ [1,3,2] -> [2,2,2] ----
    bC = NDArray.from_array([1,0,0,1,1,0], 1, 3, 2)
    broadcast_result = bA.matmul(bC)
    emit("matmul_batch_broadcast", str(broadcast_result))
    emit("matmul_batch_broadcast_shape", str(list(broadcast_result.get_shape())))

    # ---- errors ----
    try:
        NDArray.from_array([1,2,3], 3).dot(NDArray.from_array([1,2], 2))
        emit("dot_length_error", "NO_ERROR")
    except Exception:
        emit("dot_length_error", "ERROR")

    try:
        NDArray.from_array([1,2,3,4], 2, 2).matmul(NDArray.from_array([1,2,3,4,5,6], 3, 2))
        emit("matmul_dim_error", "NO_ERROR")
    except Exception:
        emit("matmul_dim_error", "ERROR")

    try:
        NDArray.from_array([1,2,3,4], 2, 2).dot(NDArray.from_array([1,2], 2))
        emit("dot_not_1d_error", "NO_ERROR")
    except Exception:
        emit("dot_not_1d_error", "ERROR")

if __name__ == "__main__":
    main()
