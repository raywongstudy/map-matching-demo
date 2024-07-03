
# 隱馬爾可夫模型及路徑匹配

## 簡介
因為最近我開始做車流預測的學習，剛需要用到 GPS 數據的路徑匹配，把不準的GPS用簡單算法做GPS點和路徑匹配, 就順便更新一個簡單專案記錄。
本專案使用隱馬爾可夫模型（Hidden Markov Model, HMM）和維特比算法實現了基於 GPS 數據的路徑匹配。隱馬爾可夫模型是一種統計模型，它被用來描述由隱藏的狀態生成觀察序列的過程。
本專案主要目的是通過給定的數據點，估計其最有可能的路徑。

## 目錄
1. [安裝](#安裝)
2. [使用方法](#使用方法)
3. [文件結構](#文件結構)
4. [貢獻](#貢獻)
5. [許可證](#許可證)

## 安裝
1. 克隆此儲存庫：
   ```bash
   git clone <儲存庫地址>
   ```
2. 安裝所需的 Python 套件：
   ```bash
   pip install <module>
   ```

## 使用方法
1. 定義隱馬爾可夫模型的參數;
![image](https://github.com/raywongstudy/map-matching-demo/blob/main/images/before.jpg)
2. 使用維特比算法估計最有可能的狀態序列：
   ```python
   # Viterbi算法
    def viterbi(pi, A, B, observations, weight_A=1, weight_B=8, weight_pi=1):
        T = len(observations)  # 观测序列的长度（點的数量）
        print("\nT:",T)
        N = A.shape[0]         # 状态的数量（线段的数量）
        print("N:",N)
        
        # 初始化动态规划表格
        V = np.zeros((N, T))   # 存储最大概率
        path = np.zeros((N, T), dtype=int)  # 存储最优路径的索引
        
        # 初始状态概率
        V[:, 0] = pi * B[0, :]  # 计算第一个观测的初始状态概率(B的概率，distance出的 % ) #pi 要由一開始定義，現用平均值0.2
        print("B[0, :] ",B[0, :]) 
        print("V[:, 0] ",V[:, 0] ,"\n\n")
        
        # 动态规划填表
        for t in range(1, T):  # 从第二个观测开始
            for s in range(N):  # 对于每个状态
                # 计算从每个可能的前一个状态到当前状态的概率
                prob = V[:, t-1] ** weight_pi * A[:, s] ** weight_A * B[t, s] ** weight_B
                print("t,s:",t,s)
                print("prob:",prob)
                # 选择最大概率并存储
                V[s, t] = np.max(prob)
                # 存储最大概率的前一个状态索引
                path[s, t] = np.argmax(prob)
        
        # 终止状态
        final_state = np.argmax(V[:, T-1])  # 找到最后一个观测的最大概率状态
        best_path = [final_state]  # 初始化最佳路径
        
        # 回溯找到最佳路径
        for t in range(T-1, 0, -1):  # 从最后一个观测往前推
            final_state = path[final_state, t]  # 找到前一个状态
            best_path.insert(0, final_state)  # 插入到路径的前面
        
        return best_path, V  # 返回最佳路径和动态规划表格

    # 觀察序列（觀察點的索引）
    observations = list(points.keys())
    print("observations:",observations)
    # 使用Viterbi算法求解
    best_path, V = viterbi(pi, A, B, observations)

   ```

3. 顯示路徑：
   ```python

    # 打印結果
    print("最佳路徑（觀察點所在的link）:", [list(lines.keys())[i] for i in best_path])

    # 繪製觀察點與最佳路徑的匹配結果
    plt.figure(figsize=(10, 6))
    for line, coords in lines.items():
        x, y = zip(*coords)
        plt.plot(x, y, marker='o', label=line)

    for point, coord in points.items():
        plt.text(coord[0], coord[1], point, fontsize=12, ha='right', color='red')

    for i, point in enumerate(observations):
        link = list(lines.keys())[best_path[i]]
        coord = points[point]
        plt.text(coord[0], coord[1], f"{point} -> {link} ->       ", fontsize=10, ha='right', color='blue')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Sample Map with Points and Viterbi Path')
    plt.legend()
    plt.grid(True)
    plt.show()

   ```
![image](https://github.com/raywongstudy/map-matching-demo/blob/main/images/after.jpg)
## 文件結構
```
map_matching/
│
├── iamge/                   # 資料夾，用於存放數據
│
├── map-matching-demo.ipynb  # 源代碼
│
└── README.md               # 本文件
```

## 貢獻
歡迎對本專案進行貢獻。您可以通過以下方式之一貢獻：
1. 提交問題（Issue）
2. 發送拉取請求（Pull Request）
3. 撰寫或改進文檔

## 許可證
本專案基於 MIT 許可證，詳見 LICENSE 文件。

---

如果有任何問題或建議，歡迎聯繫我們。
