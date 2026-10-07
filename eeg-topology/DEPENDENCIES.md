# 依赖与归属

模块使用 NumPy、MNE-Python 和 SciPy。`common.py` 调用 MNE 的标准 montage，并用 SciPy 的 `sph_harm_y`；因此需要提供该接口的 SciPy 版本。此包没有锁定或实测完整新环境。

MNE 标准电极模板不是作者新采集的数据。LuMamba、REVE、CBraMod 等研究基线也不是个人原创模型；本选集没有打包它们的代码或权重。原代码中的方法说明和上游引用保持保留。
