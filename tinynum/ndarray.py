"""N-dimensional array — the core data structure of tinynum.

Internally stores data in a flat list[float] with shape and strides
metadata, enabling zero-copy views for reshape, transpose, and slice.
"""

from __future__ import annotations
from typing import Sequence

from tinynum.dtype import DType
from tinynum.slice import Slice


class NDArray:
    """N-dimensional array with flat storage, shape, strides, and offset."""

    def __init__(self) -> None:
        self.data: list[float] = []       # flat storage
        self.shape: tuple[int, ...] = ()  # e.g. (2, 3, 4)
        self.strides: tuple[int, ...] = ()  # e.g. (12, 4, 1) row-major
        self.offset: int = 0              # for views/slices

    # ================================================================
    # S01 — Storage & Shape
    # ================================================================

    @staticmethod
    def from_array(data: list[float], *shape: int) -> "NDArray":
        """Creates an NDArray from a flat data list with the given shape.

        Raises:
            ValueError: if len(data) != product of shape
        """
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def zeros(*shape: int) -> "NDArray":
        """Creates a zero-filled NDArray with the given shape."""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def ones(*shape: int) -> "NDArray":
        """Creates a one-filled NDArray with the given shape."""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def full(value: float, *shape: int) -> "NDArray":
        """Creates an NDArray filled with value."""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def zeros_like(other: "NDArray") -> "NDArray":
        """Creates a zero-filled NDArray with the same shape as other."""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def ones_like(other: "NDArray") -> "NDArray":
        """Creates a one-filled NDArray with the same shape as other."""
        raise NotImplementedError("TODO: S01")

    def size(self) -> int:
        """Returns the total number of elements."""
        raise NotImplementedError("TODO: S01")

    def ndim(self) -> int:
        """Returns the number of dimensions."""
        raise NotImplementedError("TODO: S01")

    def get_shape(self) -> tuple[int, ...]:
        """Returns a copy of the shape tuple."""
        raise NotImplementedError("TODO: S01")

    def __str__(self) -> str:
        """Pretty-prints the NDArray: e.g. '[[1.0, 2.0], [3.0, 4.0]]'."""
        raise NotImplementedError("TODO: S01")

    def __repr__(self) -> str:
        return self.__str__()

    # ================================================================
    # S02 — Strides & Indexing
    # ================================================================

    @staticmethod
    def compute_strides(shape: Sequence[int]) -> tuple[int, ...]:
        """Computes row-major strides for the given shape.

        Example: shape (3, 4, 5) → strides (20, 5, 1)
        """
        raise NotImplementedError("TODO: S02")

    def get(self, *indices: int) -> float:
        """Gets the element at the given multi-dimensional indices.

        Uses: physical_index = offset + sum(index[i] * stride[i])
        """
        raise NotImplementedError("TODO: S02")

    def set(self, value: float, *indices: int) -> None:
        """Sets the element at the given multi-dimensional indices."""
        raise NotImplementedError("TODO: S02")

    def is_contiguous(self) -> bool:
        """Returns True if strides form a standard row-major contiguous layout."""
        raise NotImplementedError("TODO: S02")

    # ================================================================
    # S03 — Reshape
    # ================================================================

    def reshape(self, *new_shape: int) -> "NDArray":
        """Returns a view with a new shape (zero-copy when contiguous).

        Supports -1 for one dimension to auto-infer its size.
        """
        raise NotImplementedError("TODO: S03")

    def flatten(self) -> "NDArray":
        """Flattens to a 1-D array. Equivalent to reshape(-1)."""
        raise NotImplementedError("TODO: S03")

    def duplicate(self) -> "NDArray":
        """Returns a deep copy (always contiguous)."""
        raise NotImplementedError("TODO: S03")

    # ================================================================
    # S04 — Transpose
    # ================================================================

    def transpose(self, *axes: int) -> "NDArray":
        """Transpose the array.

        - No args: 2-D transpose (swap axis 0 and 1).
        - With args: N-D transpose, rearranges axes per the given permutation.

        Always zero-copy.
        """
        raise NotImplementedError("TODO: S04")

    def swap_axes(self, axis1: int, axis2: int) -> "NDArray":
        """Swaps two axes (zero-copy)."""
        raise NotImplementedError("TODO: S04")

    # ================================================================
    # S05 — Unary Math
    # ================================================================

    def neg(self) -> "NDArray":
        """Returns -x element-wise."""
        raise NotImplementedError("TODO: S05")

    def abs(self) -> "NDArray":
        """Returns |x| element-wise."""
        raise NotImplementedError("TODO: S05")

    def exp(self) -> "NDArray":
        """Returns e^x element-wise."""
        raise NotImplementedError("TODO: S05")

    def log(self) -> "NDArray":
        """Returns ln(x) element-wise."""
        raise NotImplementedError("TODO: S05")

    def sqrt(self) -> "NDArray":
        """Returns sqrt(x) element-wise."""
        raise NotImplementedError("TODO: S05")

    def square(self) -> "NDArray":
        """Returns x² element-wise."""
        raise NotImplementedError("TODO: S05")

    def tanh(self) -> "NDArray":
        """Returns tanh(x) element-wise."""
        raise NotImplementedError("TODO: S05")

    def sin(self) -> "NDArray":
        """Returns sin(x) element-wise."""
        raise NotImplementedError("TODO: S05")

    def cos(self) -> "NDArray":
        """Returns cos(x) element-wise."""
        raise NotImplementedError("TODO: S05")

    def sign(self) -> "NDArray":
        """Returns sgn(x) element-wise."""
        raise NotImplementedError("TODO: S05")

    def round(self) -> "NDArray":
        """Returns rounded values element-wise."""
        raise NotImplementedError("TODO: S05")

    def clip(self, min_val: float, max_val: float) -> "NDArray":
        """Clips values to [min_val, max_val] element-wise."""
        raise NotImplementedError("TODO: S05")

    def pow(self, p: float) -> "NDArray":
        """Returns x^p element-wise."""
        raise NotImplementedError("TODO: S05")

    # ================================================================
    # S06 — Binary Ops & Comparisons (same shape)
    # ================================================================

    # --- Arithmetic (NDArray) ---

    def add(self, other: "NDArray") -> "NDArray":
        """Element-wise addition."""
        raise NotImplementedError("TODO: S06")

    def sub(self, other: "NDArray") -> "NDArray":
        """Element-wise subtraction."""
        raise NotImplementedError("TODO: S06")

    def mul(self, other: "NDArray") -> "NDArray":
        """Element-wise multiplication."""
        raise NotImplementedError("TODO: S06")

    def div(self, other: "NDArray") -> "NDArray":
        """Element-wise division."""
        raise NotImplementedError("TODO: S06")

    def pow_array(self, other: "NDArray") -> "NDArray":
        """Element-wise power: x^y."""
        raise NotImplementedError("TODO: S06")

    def maximum(self, other: "NDArray") -> "NDArray":
        """Element-wise maximum: max(x, y)."""
        raise NotImplementedError("TODO: S06")

    # --- Arithmetic (scalar) ---

    def add_scalar(self, scalar: float) -> "NDArray":
        """Adds a scalar to every element."""
        raise NotImplementedError("TODO: S06")

    def sub_scalar(self, scalar: float) -> "NDArray":
        """Subtracts a scalar from every element."""
        raise NotImplementedError("TODO: S06")

    def mul_scalar(self, scalar: float) -> "NDArray":
        """Multiplies every element by a scalar."""
        raise NotImplementedError("TODO: S06")

    def div_scalar(self, scalar: float) -> "NDArray":
        """Divides every element by a scalar."""
        raise NotImplementedError("TODO: S06")

    # --- Comparisons (NDArray) — returns 1.0 / 0.0 ---

    def eq(self, other: "NDArray") -> "NDArray":
        """Element-wise equal: returns 1.0 where x == y."""
        raise NotImplementedError("TODO: S06")

    def neq(self, other: "NDArray") -> "NDArray":
        """Element-wise not-equal: returns 1.0 where x != y."""
        raise NotImplementedError("TODO: S06")

    def gt(self, other: "NDArray") -> "NDArray":
        """Element-wise greater-than: returns 1.0 where x > y."""
        raise NotImplementedError("TODO: S06")

    def gte(self, other: "NDArray") -> "NDArray":
        """Element-wise greater-than-or-equal: returns 1.0 where x >= y."""
        raise NotImplementedError("TODO: S06")

    def lt(self, other: "NDArray") -> "NDArray":
        """Element-wise less-than: returns 1.0 where x < y."""
        raise NotImplementedError("TODO: S06")

    def lte(self, other: "NDArray") -> "NDArray":
        """Element-wise less-than-or-equal: returns 1.0 where x <= y."""
        raise NotImplementedError("TODO: S06")

    # --- Comparisons (scalar) ---

    def eq_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x == scalar."""
        raise NotImplementedError("TODO: S06")

    def neq_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x != scalar."""
        raise NotImplementedError("TODO: S06")

    def gt_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x > scalar."""
        raise NotImplementedError("TODO: S06")

    def gte_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x >= scalar."""
        raise NotImplementedError("TODO: S06")

    def lt_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x < scalar."""
        raise NotImplementedError("TODO: S06")

    def lte_scalar(self, scalar: float) -> "NDArray":
        """Returns 1.0 where x <= scalar."""
        raise NotImplementedError("TODO: S06")

    # ================================================================
    # S07 — Broadcasting
    # ================================================================

    @staticmethod
    def broadcast_shapes(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple[int, ...]:
        """Computes the broadcast-compatible output shape.

        Example: (3, 1) + (1, 4) → (3, 4)

        Raises:
            ValueError: if shapes are not broadcast-compatible
        """
        raise NotImplementedError("TODO: S07")

    def broadcast_to(self, *target_shape: int) -> "NDArray":
        """Returns a view broadcast to the target shape (zero-copy, stride=0 trick)."""
        raise NotImplementedError("TODO: S07")

    # ================================================================
    # S08 — Reduction: Sum & Mean
    # ================================================================

    def sum_all(self) -> float:
        """Returns the sum of all elements."""
        raise NotImplementedError("TODO: S08")

    def mean_all(self) -> float:
        """Returns the mean of all elements."""
        raise NotImplementedError("TODO: S08")

    def sum(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Sum along a single axis."""
        raise NotImplementedError("TODO: S08")

    def mean(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Mean along a single axis."""
        raise NotImplementedError("TODO: S08")

    def sum_axes(self, axes: Sequence[int], keep_dims: bool = False) -> "NDArray":
        """Sum along multiple axes."""
        raise NotImplementedError("TODO: S08")

    # ================================================================
    # S09 — Reduction: Max, Var & friends
    # ================================================================

    def max(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Max along an axis."""
        raise NotImplementedError("TODO: S09")

    def min(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Min along an axis."""
        raise NotImplementedError("TODO: S09")

    def argmax(self, axis: int) -> "NDArray":
        """Index of the maximum value along an axis."""
        raise NotImplementedError("TODO: S09")

    def argmin(self, axis: int) -> "NDArray":
        """Index of the minimum value along an axis."""
        raise NotImplementedError("TODO: S09")

    def prod(self, axis: int) -> "NDArray":
        """Product of elements along an axis."""
        raise NotImplementedError("TODO: S09")

    def var(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Variance along an axis."""
        raise NotImplementedError("TODO: S09")

    def std(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """Standard deviation along an axis."""
        raise NotImplementedError("TODO: S09")

    def count_nonzero(self) -> int:
        """Counts non-zero elements."""
        raise NotImplementedError("TODO: S09")

    # ================================================================
    # S10 — MatMul
    # ================================================================

    def dot(self, other: "NDArray") -> "NDArray":
        """Vector dot product (1-D · 1-D → scalar wrapped in 0-D array)."""
        raise NotImplementedError("TODO: S10")

    def matmul(self, other: "NDArray") -> "NDArray":
        """Matrix multiplication.

        - 2-D × 2-D: (M,K) × (K,N) → (M,N)
        - Batched: (...,M,K) × (...,K,N) → (...,M,N)
        """
        raise NotImplementedError("TODO: S10")

    # ================================================================
    # S11 — Slicing & Views
    # ================================================================

    def slice(self, *ranges: Slice) -> "NDArray":
        """Returns a view into a sub-region of this array (zero-copy).

        Args:
            ranges: one Slice per axis
        """
        raise NotImplementedError("TODO: S11")

    def expand_dims(self, axis: int) -> "NDArray":
        """Adds a dimension of size 1 at the given axis."""
        raise NotImplementedError("TODO: S11")

    def squeeze_axis(self, axis: int) -> "NDArray":
        """Removes a dimension of size 1 at the given axis."""
        raise NotImplementedError("TODO: S11")

    def squeeze(self) -> "NDArray":
        """Removes all dimensions of size 1."""
        raise NotImplementedError("TODO: S11")

    # ================================================================
    # S12 — Creation & Random
    # ================================================================

    @staticmethod
    def arange(start: float, end: float, step: float) -> "NDArray":
        """Creates a 1-D array: [start, start+step, ..., end)."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def linspace(start: float, end: float, num: int) -> "NDArray":
        """Creates a 1-D array of num evenly spaced values in [start, end]."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def eye(n: int) -> "NDArray":
        """Creates an n×n identity matrix."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def diag(vector: "NDArray") -> "NDArray":
        """Creates a diagonal matrix from a 1-D vector."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def randn(*shape: int) -> "NDArray":
        """Creates an NDArray with standard normal random values N(0,1)."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def rand(*shape: int) -> "NDArray":
        """Creates an NDArray with uniform random values in [0, 1)."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def uniform(lo: float, hi: float, *shape: int) -> "NDArray":
        """Creates an NDArray with uniform random values in [lo, hi)."""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def shuffle(indices: list[int]) -> None:
        """Shuffles an index list in-place (Fisher-Yates)."""
        raise NotImplementedError("TODO: S12")

    def fill(self, value: float) -> None:
        """Fills all elements with the given value (in-place)."""
        raise NotImplementedError("TODO: S12")

    # ================================================================
    # S13 — Join & Transform
    # ================================================================

    @staticmethod
    def concatenate(arrays: list["NDArray"], axis: int) -> "NDArray":
        """Concatenates arrays along an existing axis."""
        raise NotImplementedError("TODO: S13")

    @staticmethod
    def stack(arrays: list["NDArray"], axis: int) -> "NDArray":
        """Stacks arrays along a new axis."""
        raise NotImplementedError("TODO: S13")

    def pad(self, pad_width: list[tuple[int, int]], value: float = 0.0) -> "NDArray":
        """Pads this array.

        Args:
            pad_width: [(before, after)] for each axis
            value: fill value for padded regions
        """
        raise NotImplementedError("TODO: S13")

    def flip(self, axis: int) -> "NDArray":
        """Reverses elements along the given axis."""
        raise NotImplementedError("TODO: S13")

    # ================================================================
    # S14 — Fancy Indexing
    # ================================================================

    def index_select(self, axis: int, indices: list[int]) -> "NDArray":
        """Selects rows/columns by index (embedding lookup).

        Example: weight.index_select(0, [3, 0, 3, 7])
        """
        raise NotImplementedError("TODO: S14")

    def scatter_add(self, axis: int, indices: list[int], src: "NDArray") -> None:
        """Scatter-adds src into this array (embedding backward).

        Equivalent to numpy's np.add.at(self, indices, src).
        """
        raise NotImplementedError("TODO: S14")

    def masked_fill(self, mask: "NDArray", value: float) -> "NDArray":
        """Returns a new array where positions with mask == 1.0 are replaced by value."""
        raise NotImplementedError("TODO: S14")

    @staticmethod
    def where(condition: "NDArray", x: "NDArray", y: "NDArray") -> "NDArray":
        """Element-wise conditional: picks from x where condition is non-zero, else from y."""
        raise NotImplementedError("TODO: S14")

    # ================================================================
    # S15 — Capstone: Toolkit
    # ================================================================

    @staticmethod
    def tril(n: int, diagonal: int = 0) -> "NDArray":
        """Lower-triangular matrix of size n (for causal masks)."""
        raise NotImplementedError("TODO: S15")

    @staticmethod
    def triu(n: int, diagonal: int = 0) -> "NDArray":
        """Upper-triangular matrix of size n."""
        raise NotImplementedError("TODO: S15")

    def norm(self, axis: int) -> "NDArray":
        """L2 norm along an axis."""
        raise NotImplementedError("TODO: S15")

    def diff(self, axis: int) -> "NDArray":
        """Differences between consecutive elements along an axis."""
        raise NotImplementedError("TODO: S15")

    def percentile(self, q: float) -> "NDArray":
        """Computes the q-th percentile across all elements."""
        raise NotImplementedError("TODO: S15")

    def argsort(self, axis: int) -> "NDArray":
        """Returns the indices that would sort along the given axis."""
        raise NotImplementedError("TODO: S15")

    def unique(self) -> "NDArray":
        """Returns sorted unique elements."""
        raise NotImplementedError("TODO: S15")

    def all_close(self, other: "NDArray", atol: float = 1e-6) -> bool:
        """Returns True if all elements are within atol of other."""
        raise NotImplementedError("TODO: S15")

    def astype(self, dtype: DType) -> "NDArray":
        """Converts element type: float32 ↔ int8."""
        raise NotImplementedError("TODO: S15")
