# 上游归属与依赖

冻结主干：Physical Intelligence 的 [OpenPI](https://github.com/Physical-Intelligence/openpi)，代码内固定 commit 为 `215abfb217dbac7d5f1273282331b9b1866c0479`。

评估环境：[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)，代码内固定 commit 为 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。

π0.5 模型、检查点、OpenPI 与 LIBERO 本身均为上游成果。本包仅含本地评估包装器和计时工具，没有复制第三方整库、权重或数据，原上游许可继续适用。代码中的版本核对不等于本次已重跑该冻结基线。

完整环境涉及 Python、JAX、Orbax、图像处理与 MuJoCo / EGL 等依赖，原工作区分离模型服务和仿真客户端环境。本选集未生成或验证新的统一锁文件。
