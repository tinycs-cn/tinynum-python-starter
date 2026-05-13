"""N 维数组 —— tinynum 的核心数据结构。

内部使用连续的 list[float] 存储数据，配合 shape 和 strides 元数据，
支持 reshape、transpose、slice 的零拷贝视图。
"""

from __future__ import annotations
from typing import Sequence

from tinynum.dtype import DType
from tinynum.slice import Slice


class NDArray:
    """一维连续存储 + shape/strides 元数据的 N 维数组。"""

    def __init__(self) -> None:
        self.data: list[float] = []       # 一维连续存储
        self.shape: tuple[int, ...] = ()  # 各维大小，如 (2, 3, 4)
        self.strides: tuple[int, ...] = ()  # 各维步长，如 (12, 4, 1)（行优先）
        self.offset: int = 0              # 视图/切片的起始偏移

    # ================================================================
    # S01 — 存储与形状
    # ================================================================

    @staticmethod
    def from_array(data: list[float], *shape: int) -> "NDArray":
        """从一维数据列表和指定 shape 创建 NDArray。

        S01 只需初始化 data 和 shape；
        strides 在 S02 中初始化，offset 默认为 0。

        Raises:
            ValueError: 若 len(data) 不等于 shape 各维之积
        """
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def zeros(*shape: int) -> "NDArray":
        """创建全零 NDArray。"""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def ones(*shape: int) -> "NDArray":
        """创建全一 NDArray。"""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def full(value: float, *shape: int) -> "NDArray":
        """创建以 value 填充的 NDArray。"""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def zeros_like(other: "NDArray") -> "NDArray":
        """创建与 other 同形的全零 NDArray。"""
        raise NotImplementedError("TODO: S01")

    @staticmethod
    def ones_like(other: "NDArray") -> "NDArray":
        """创建与 other 同形的全一 NDArray。"""
        raise NotImplementedError("TODO: S01")

    def size(self) -> int:
        """返回元素总数。"""
        raise NotImplementedError("TODO: S01")

    def ndim(self) -> int:
        """返回维度数。"""
        raise NotImplementedError("TODO: S01")

    def get_shape(self) -> tuple[int, ...]:
        """返回 shape 元组的副本。"""
        raise NotImplementedError("TODO: S01")

    def __str__(self) -> str:
        """格式化打印，如 '[[1.0, 2.0], [3.0, 4.0]]'。"""
        raise NotImplementedError("TODO: S01")

    def __repr__(self) -> str:
        return self.__str__()

    # ================================================================
    # S02 — 步长与索引
    # ================================================================

    @staticmethod
    def compute_strides(shape: Sequence[int]) -> tuple[int, ...]:
        """计算行优先步长。

        示例：shape (3, 4, 5) → strides (20, 5, 1)
        """
        raise NotImplementedError("TODO: S02")

    def get(self, *indices: int) -> float:
        """按多维索引取值。

        物理索引 = offset + sum(index[i] * stride[i])
        """
        raise NotImplementedError("TODO: S02")

    def set(self, value: float, *indices: int) -> None:
        """按多维索引赋值。"""
        raise NotImplementedError("TODO: S02")

    def is_contiguous(self) -> bool:
        """判断是否为标准行优先连续布局。"""
        raise NotImplementedError("TODO: S02")

    # ================================================================
    # S03 — 变形
    # ================================================================

    def reshape(self, *new_shape: int) -> "NDArray":
        """返回指定形状的视图（连续时零拷贝）。

        支持一个维度传入 -1 自动推断大小。
        """
        raise NotImplementedError("TODO: S03")

    def flatten(self) -> "NDArray":
        """展平为一维数组，等价于 reshape(-1)。"""
        raise NotImplementedError("TODO: S03")

    def duplicate(self) -> "NDArray":
        """返回深拷贝（始终连续）。"""
        raise NotImplementedError("TODO: S03")

    # ================================================================
    # S04 — 转置
    # ================================================================

    def transpose(self, *axes: int) -> "NDArray":
        """转置数组。

        - 无参数：二维转置（交换 axis 0 与 axis 1）。
        - 有参数：N 维转置，按给定排列重排轴顺序。

        始终零拷贝。
        """
        raise NotImplementedError("TODO: S04")

    def swap_axes(self, axis1: int, axis2: int) -> "NDArray":
        """交换两个轴（零拷贝）。"""
        raise NotImplementedError("TODO: S04")

    # ================================================================
    # S05 — 一元运算
    # ================================================================

    def neg(self) -> "NDArray":
        """逐元素取负。"""
        raise NotImplementedError("TODO: S05")

    def abs(self) -> "NDArray":
        """逐元素取绝对值。"""
        raise NotImplementedError("TODO: S05")

    def exp(self) -> "NDArray":
        """逐元素计算 e^x。"""
        raise NotImplementedError("TODO: S05")

    def log(self) -> "NDArray":
        """逐元素计算自然对数。"""
        raise NotImplementedError("TODO: S05")

    def sqrt(self) -> "NDArray":
        """逐元素计算平方根。"""
        raise NotImplementedError("TODO: S05")

    def square(self) -> "NDArray":
        """逐元素计算平方。"""
        raise NotImplementedError("TODO: S05")

    def tanh(self) -> "NDArray":
        """逐元素计算 tanh。"""
        raise NotImplementedError("TODO: S05")

    def sin(self) -> "NDArray":
        """逐元素计算 sin。"""
        raise NotImplementedError("TODO: S05")

    def cos(self) -> "NDArray":
        """逐元素计算 cos。"""
        raise NotImplementedError("TODO: S05")

    def sign(self) -> "NDArray":
        """逐元素取符号（sgn）。"""
        raise NotImplementedError("TODO: S05")

    def round(self) -> "NDArray":
        """逐元素四舍五入。"""
        raise NotImplementedError("TODO: S05")

    def clip(self, min_val: float, max_val: float) -> "NDArray":
        """逐元素截断到 [min_val, max_val]。"""
        raise NotImplementedError("TODO: S05")

    def pow(self, p: float) -> "NDArray":
        """逐元素计算 x^p。"""
        raise NotImplementedError("TODO: S05")

    # ================================================================
    # S06 — 二元运算与比较（同形）
    # ================================================================

    # --- 数组间算术 ---

    def add(self, other: "NDArray") -> "NDArray":
        """逐元素加法。"""
        raise NotImplementedError("TODO: S06")

    def sub(self, other: "NDArray") -> "NDArray":
        """逐元素减法。"""
        raise NotImplementedError("TODO: S06")

    def mul(self, other: "NDArray") -> "NDArray":
        """逐元素乘法。"""
        raise NotImplementedError("TODO: S06")

    def div(self, other: "NDArray") -> "NDArray":
        """逐元素除法。"""
        raise NotImplementedError("TODO: S06")

    def pow_array(self, other: "NDArray") -> "NDArray":
        """逐元素幂运算：x^y。"""
        raise NotImplementedError("TODO: S06")

    def maximum(self, other: "NDArray") -> "NDArray":
        """逐元素取最大值。"""
        raise NotImplementedError("TODO: S06")

    # --- 标量算术 ---

    def add_scalar(self, scalar: float) -> "NDArray":
        """每个元素加标量。"""
        raise NotImplementedError("TODO: S06")

    def sub_scalar(self, scalar: float) -> "NDArray":
        """每个元素减标量。"""
        raise NotImplementedError("TODO: S06")

    def mul_scalar(self, scalar: float) -> "NDArray":
        """每个元素乘标量。"""
        raise NotImplementedError("TODO: S06")

    def div_scalar(self, scalar: float) -> "NDArray":
        """每个元素除以标量。"""
        raise NotImplementedError("TODO: S06")

    # --- 数组间比较：结果为 1.0 / 0.0 ---

    def eq(self, other: "NDArray") -> "NDArray":
        """逐元素相等，满足处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def neq(self, other: "NDArray") -> "NDArray":
        """逐元素不等，满足处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def gt(self, other: "NDArray") -> "NDArray":
        """逐元素大于，满足处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def gte(self, other: "NDArray") -> "NDArray":
        """逐元素大于等于，满足处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def lt(self, other: "NDArray") -> "NDArray":
        """逐元素小于，满足处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def lte(self, other: "NDArray") -> "NDArray":
        """逐元素小于等于，满足处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    # --- 标量比较 ---

    def eq_scalar(self, scalar: float) -> "NDArray":
        """x == scalar 处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def neq_scalar(self, scalar: float) -> "NDArray":
        """x != scalar 处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def gt_scalar(self, scalar: float) -> "NDArray":
        """x > scalar 处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def gte_scalar(self, scalar: float) -> "NDArray":
        """x >= scalar 处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def lt_scalar(self, scalar: float) -> "NDArray":
        """x < scalar 处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    def lte_scalar(self, scalar: float) -> "NDArray":
        """x <= scalar 处返回 1.0。"""
        raise NotImplementedError("TODO: S06")

    # ================================================================
    # S07 — 广播
    # ================================================================

    @staticmethod
    def broadcast_shapes(shape_a: Sequence[int], shape_b: Sequence[int]) -> tuple[int, ...]:
        """计算两个 shape 广播后的输出 shape。

        示例：(3, 1) + (1, 4) → (3, 4)

        Raises:
            ValueError: 若两个 shape 不兼容广播
        """
        raise NotImplementedError("TODO: S07")

    def broadcast_to(self, *target_shape: int) -> "NDArray":
        """返回广播到目标 shape 的视图（零拷贝，利用 stride=0 技巧）。"""
        raise NotImplementedError("TODO: S07")

    # ================================================================
    # S08 — 归约：求和与均值
    # ================================================================

    def sum_all(self) -> float:
        """所有元素求和。"""
        raise NotImplementedError("TODO: S08")

    def mean_all(self) -> float:
        """所有元素求均值。"""
        raise NotImplementedError("TODO: S08")

    def sum(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """沿指定轴求和。"""
        raise NotImplementedError("TODO: S08")

    def mean(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """沿指定轴求均值。"""
        raise NotImplementedError("TODO: S08")

    def sum_axes(self, axes: Sequence[int], keep_dims: bool = False) -> "NDArray":
        """沿多个轴求和。"""
        raise NotImplementedError("TODO: S08")

    # ================================================================
    # S09 — 归约：最大值、方差等
    # ================================================================

    def max(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """沿指定轴取最大值。"""
        raise NotImplementedError("TODO: S09")

    def min(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """沿指定轴取最小值。"""
        raise NotImplementedError("TODO: S09")

    def argmax(self, axis: int) -> "NDArray":
        """沿指定轴返回最大值的索引。"""
        raise NotImplementedError("TODO: S09")

    def argmin(self, axis: int) -> "NDArray":
        """沿指定轴返回最小值的索引。"""
        raise NotImplementedError("TODO: S09")

    def prod(self, axis: int) -> "NDArray":
        """沿指定轴求乘积。"""
        raise NotImplementedError("TODO: S09")

    def var(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """沿指定轴计算方差。"""
        raise NotImplementedError("TODO: S09")

    def std(self, axis: int, keep_dims: bool = False) -> "NDArray":
        """沿指定轴计算标准差。"""
        raise NotImplementedError("TODO: S09")

    def count_nonzero(self) -> int:
        """统计非零元素个数。"""
        raise NotImplementedError("TODO: S09")

    # ================================================================
    # S10 — 矩阵乘法
    # ================================================================

    def dot(self, other: "NDArray") -> "NDArray":
        """向量点积（1-D · 1-D → 标量，包装为 0-D 数组）。"""
        raise NotImplementedError("TODO: S10")

    def matmul(self, other: "NDArray") -> "NDArray":
        """矩阵乘法。

        - 二维：(M,K) × (K,N) → (M,N)
        - 批量：(...,M,K) × (...,K,N) → (...,M,N)
        """
        raise NotImplementedError("TODO: S10")

    # ================================================================
    # S11 — 切片与视图
    # ================================================================

    def slice(self, *ranges: Slice) -> "NDArray":
        """返回子区域视图（零拷贝）。

        Args:
            ranges: 每个轴对应一个 Slice
        """
        raise NotImplementedError("TODO: S11")

    def expand_dims(self, axis: int) -> "NDArray":
        """在指定轴插入大小为 1 的新维度。"""
        raise NotImplementedError("TODO: S11")

    def squeeze_axis(self, axis: int) -> "NDArray":
        """移除指定轴上大小为 1 的维度。"""
        raise NotImplementedError("TODO: S11")

    def squeeze(self) -> "NDArray":
        """移除所有大小为 1 的维度。"""
        raise NotImplementedError("TODO: S11")

    # ================================================================
    # S12 — 创建与随机
    # ================================================================

    @staticmethod
    def arange(start: float, end: float, step: float) -> "NDArray":
        """创建一维等差数组：[start, end)，步长为 step。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def linspace(start: float, end: float, num: int) -> "NDArray":
        """在 [start, end] 内均匀生成 num 个值。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def eye(n: int) -> "NDArray":
        """创建 n×n 单位矩阵。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def diag(vector: "NDArray") -> "NDArray":
        """以一维向量为对角线创建对角矩阵。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def randn(*shape: int) -> "NDArray":
        """创建标准正态分布随机数组 N(0,1)。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def rand(*shape: int) -> "NDArray":
        """创建 [0, 1) 均匀分布随机数组。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def uniform(lo: float, hi: float, *shape: int) -> "NDArray":
        """创建 [lo, hi) 均匀分布随机数组。"""
        raise NotImplementedError("TODO: S12")

    @staticmethod
    def shuffle(indices: list[int]) -> None:
        """原地随机打乱索引列表（Fisher-Yates 算法）。"""
        raise NotImplementedError("TODO: S12")

    def fill(self, value: float) -> None:
        """原地将所有元素填充为指定值。"""
        raise NotImplementedError("TODO: S12")

    # ================================================================
    # S13 — 拼接与变换
    # ================================================================

    @staticmethod
    def concatenate(arrays: list["NDArray"], axis: int) -> "NDArray":
        """沿已有轴拼接数组。"""
        raise NotImplementedError("TODO: S13")

    @staticmethod
    def stack(arrays: list["NDArray"], axis: int) -> "NDArray":
        """沿新轴堆叠数组。"""
        raise NotImplementedError("TODO: S13")

    def pad(self, pad_width: list[tuple[int, int]], value: float = 0.0) -> "NDArray":
        """对数组进行填充。

        Args:
            pad_width: [(前, 后)] 表示每个轴两侧的填充量
            value: 填充值
        """
        raise NotImplementedError("TODO: S13")

    def flip(self, axis: int) -> "NDArray":
        """沿指定轴翻转元素。"""
        raise NotImplementedError("TODO: S13")

    # ================================================================
    # S14 — 花式索引
    # ================================================================

    def index_select(self, axis: int, indices: list[int]) -> "NDArray":
        """按索引选取行/列（用于 Embedding 查表）。

        示例：weight.index_select(0, [3, 0, 3, 7])
        """
        raise NotImplementedError("TODO: S14")

    def scatter_add(self, axis: int, indices: list[int], src: "NDArray") -> None:
        """将 src 按索引散射累加到本数组（Embedding 反向传播）。

        等价于 numpy 的 np.add.at(self, indices, src)。
        """
        raise NotImplementedError("TODO: S14")

    def masked_fill(self, mask: "NDArray", value: float) -> "NDArray":
        """返回新数组，mask == 1.0 的位置替换为 value。"""
        raise NotImplementedError("TODO: S14")

    @staticmethod
    def where(condition: "NDArray", x: "NDArray", y: "NDArray") -> "NDArray":
        """逐元素条件选择：condition 非零取 x，否则取 y。"""
        raise NotImplementedError("TODO: S14")

    # ================================================================
    # S15 — 收官：工具箱
    # ================================================================

    @staticmethod
    def tril(n: int, diagonal: int = 0) -> "NDArray":
        """生成 n×n 下三角矩阵（用于因果掩码）。"""
        raise NotImplementedError("TODO: S15")

    @staticmethod
    def triu(n: int, diagonal: int = 0) -> "NDArray":
        """生成 n×n 上三角矩阵。"""
        raise NotImplementedError("TODO: S15")

    def norm(self, axis: int) -> "NDArray":
        """沿指定轴计算 L2 范数。"""
        raise NotImplementedError("TODO: S15")

    def diff(self, axis: int) -> "NDArray":
        """沿指定轴计算相邻元素差分。"""
        raise NotImplementedError("TODO: S15")

    def percentile(self, q: float) -> "NDArray":
        """计算所有元素的第 q 百分位数。"""
        raise NotImplementedError("TODO: S15")

    def argsort(self, axis: int) -> "NDArray":
        """返回沿指定轴排序所需的索引。"""
        raise NotImplementedError("TODO: S15")

    def unique(self) -> "NDArray":
        """返回去重后的排序元素。"""
        raise NotImplementedError("TODO: S15")

    def all_close(self, other: "NDArray", atol: float = 1e-6) -> bool:
        """若所有元素与 other 对应元素之差均在 atol 以内，则返回 True。"""
        raise NotImplementedError("TODO: S15")

    def astype(self, dtype: DType) -> "NDArray":
        """转换元素类型：float32 ↔ int8。"""
        raise NotImplementedError("TODO: S15")
