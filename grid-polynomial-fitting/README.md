# 多项式拟合与形变测量 / Grid Polynomial Fitting

2025—2026 学年秋季《计算思维实训1》课程工程。本人署名报告题目为“多项式线性回归在测量物体形变中的简单尝试”。

使用 C++ 实现最小二乘多项式拟合、高斯消元与网格线计算，Python 绘图。原报告将 YOLOv8-OBB 网格角点与后续拟合结合。此目录保留原始文件名 `mian.cpp` 和 `leaner_regression_solve.h`。

```sh
g++ -std=c++17 mian.cpp -o fit
./fit
python draw_lines.py
```

`data.txt` 是本地保留的数值输入；运行绘图脚本前必须自行提供 `cropped_256x256.png` 背景图。课程报告和学号信息未发布。本次未将课程报告中的结果作为独立复现实验结果。
