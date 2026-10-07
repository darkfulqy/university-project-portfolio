# 深度学习基础练习

2026 年夏季本地练习留档，用于记录单神经元、梯度计算、激活函数和 GPU 编程的学习过程。这些文件含练习题干和框架，不作为完整机器学习库展示。

| 文件 | 本地留档状态 |
| --- | --- |
| `sigmond_pytorch.py` | 包含 sigmoid 计算实现和示例 |
| `softmax_pytorch.py` | 使用 PyTorch softmax 的练习实现 |
| `simple_neuro.py` | 单神经元前向计算与 MSE |
| `simple_neuro_with_backpropagation.py` | 单神经元手工梯度及 SGD 更新练习 |
| `dense_layer.py` | **未完成**：关键方法仍为 `pass` |
| `simple_autograd.py` | **未完成**：接口和算子实现不完整，存在属性赋值及构造调用问题 |
| `softmax_cuda.cpp` | **未完成**：CUDA 内核与宿主流程仍是 TODO |

成长报告对全连接层、最小自动求导与 CUDA softmax 的完成度概括偏强。本次依据代码把它们标注为练习框架，未补写实现或改变原始学习记录。

已填写的 Python 练习依赖 PyTorch，`dense_layer.py` 还导入 NumPy。本次只做静态检查，未执行这些文件，不能据此声称全部练习可运行或已经验证。来源为本地 `deep-ml` 文件夹与成长报告；未包含环境、下载资料或凭证。
