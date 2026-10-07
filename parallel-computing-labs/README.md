# OpenMP and CUDA Coursework

上海大学 2025—2026 学年春季《计算机系统结构》课程实验的精选源码与历史结果。李卿阳署名的 OpenMP、CUDA 报告及课程平台提交记录保存在本地证据档案中。本目录是课程实现的归档，不主张算法本身的原创性。

## Contents

- `juzhenchengfa.cpp`：整数矩阵乘法，比较 OpenMP static、dynamic、guided 调度。
- `mandelbrot_openmp.cpp`：曼德勃罗集合并行计算。
- `pi_openmp.cpp`：数值积分与 OpenMP reduction 计算 π。
- `matrix_cuda.cu`：使用二维线程块和共享内存的 CUDA 矩阵乘法。
- `run_cuda_experiment.ps1`：原 Windows 编译、运行和 CSV 采集脚本。
- `historical-results/`：原工程保留的数值测试表；使用生成的矩阵和实验参数，不包含个人、客户或业务数据。

## Build prerequisites

OpenMP 文件需要支持 OpenMP 的 C++ 编译器，例如：

```sh
g++ -std=c++17 -O2 -fopenmp juzhenchengfa.cpp -o matrix_openmp
g++ -std=c++17 -O2 -fopenmp mandelbrot_openmp.cpp -o mandelbrot_openmp
g++ -std=c++17 -O2 -fopenmp pi_openmp.cpp -o pi_openmp
```

CUDA 文件需要 NVIDIA GPU、驱动与兼容的 CUDA Toolkit/宿主编译器。原 PowerShell 脚本写定 Visual Studio 2022 Build Tools 的常见安装位置及 `sm_86` 架构；运行前须按机器调整。脚本会创建 `output/`，并在源码目录写入或覆盖新的性能 CSV，历史表另存于 `historical-results/`。

## Evidence and limitations

源码和历史 CSV 均按原文件复制，没有重写算法。原 CUDA 报告留痕为 2026 年 5 月；课程归档标为 2026 春。本次整理只做静态检查，未编译、运行或独立复现实验。历史加速比与具体 CPU 实现、GPU、矩阵规模和计时范围有关，不代表普遍性能提升。报告中的校验和值不是逐元素误差证明。

本公开目录不含学号、课程报告、教师模板、虚拟机镜像、编译结果或原始账号配置。未另行添加覆盖原课程材料和第三方内容的许可。
