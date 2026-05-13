"""NDArray 支持的数据类型。

课程全程使用 FLOAT32。INT8 在 S15 引入，
为 tinytorch 的 int8 量化做准备。
"""

from enum import Enum


class DType(Enum):
    FLOAT32 = "float32"
    INT8 = "int8"
