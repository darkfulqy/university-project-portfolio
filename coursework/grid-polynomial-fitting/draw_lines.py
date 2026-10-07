import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np

bg = cv2.imread("cropped_256x256.png")

bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

h, w, _ = bg.shape

plt.figure(figsize=(6, 6))
plt.imshow(bg_rgb)
plt.title("fitted grid")
plt.axis('off')

# 绘制横线拟合曲线
for f in sorted(glob.glob("row_fit_*.txt")):
    data = np.loadtxt(f)
    if data.ndim == 1:  
        data = data.reshape(1, -1)
    plt.plot(data[:, 0], data[:, 1], color='cyan', linewidth=1.5, label='row' if f == "row_fit_0.txt" else "")

# 绘制竖线拟合曲线
for f in sorted(glob.glob("col_fit_*.txt")):
    data = np.loadtxt(f)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    plt.plot(data[:, 0], data[:, 1], color='magenta', linewidth=1.5, label='col' if f == "col_fit_0.txt" else "")

plt.xlim(0, w)
plt.ylim(h, 0)
plt.legend()
plt.tight_layout()
plt.savefig("fit_overlay.png", dpi=300)
plt.show()
