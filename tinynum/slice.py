"""NDArray 索引的切片描述器。

类似 Python 内置 slice，但为 tinynum 的
NDArray.slice() 方法显式设计。
"""

from __future__ import annotations
from dataclasses import dataclass
import sys


@dataclass(frozen=True)
class Slice:
    """描述单个轴的切片范围：[start, stop)，步长为 step。"""

    start: int
    stop: int
    step: int = 1

    @staticmethod
    def of(start: int, stop: int, step: int = 1) -> "Slice":
        """创建指定步长的切片 [start, stop)。"""
        return Slice(start, stop, step)

    @staticmethod
    def all() -> "Slice":
        """选取整个轴（等价于 Python 的 ':'）。"""
        return Slice(0, sys.maxsize, 1)
