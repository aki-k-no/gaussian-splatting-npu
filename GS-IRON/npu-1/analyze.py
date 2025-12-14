import csv
import matplotlib.pyplot as plt
import numpy as np

# CSV 読み込み
filename = "data.csv"
data = []

with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        for val in row:
            try:
                data.append(float(val))
            except ValueError:
                continue  # 数値以外はスキップ


# NumPy 配列に変換
data = np.array(data)

# 有限値のみ抽出
data = data[np.isfinite(data)]
# ヒストグラム表示


vmin = 0.98
vmax = 1.02

plt.figure(figsize=(10,6))
plt.hist(data, bins=50, range=(vmin, vmax), color='skyblue', edgecolor='black')
plt.title("Deviation from correct value")
plt.xlabel("Value")
plt.ylabel("Count")
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("histogram.png")