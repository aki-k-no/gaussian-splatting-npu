import csv
import matplotlib.pyplot as plt
import numpy as np

# CSV 読み込み
for i in range(6):
    filename = f"csv/cov3_diff_data_chair_{i}.csv"
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
    data = data[~np.isnan(data)]
    # ヒストグラム表示


    mean_val = float(np.mean(data))
    # フィルタリング: 絶対値が10を超える要素を除外
    # data = data[np.abs(data - mean_val) <= 1]

    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    print("mean", mean_val)
    print("std", std_val)

    #count
    cnt = data.size
    print("size", cnt)



    vmin = 0.92
    vmax = 1.08

    plt.figure(figsize=(10,6))
    plt.hist(data, bins=50, range=(vmin, vmax), color='skyblue', edgecolor='black')
    plt.title("Deviation from correct value")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f"csv/histogram_chair_{i}.png")