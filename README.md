# MediaPipe 使用 Python 開發

> **最後更新**: 2025年12月31日

這是一個以「教學／自學」為導向的 MediaPipe Python 範例集合，使用 **MediaPipe Tasks + OpenCV** 完成：

- 環境檢測（版本、GPU/CUDA 等）
- 物件偵測（單張圖片、USB 攝影機串流）
- 手部地標偵測（USB 攝影機串流）

## 📌 目錄

- [快速開始（建議先做）](#-快速開始建議先做)
- [學習路徑（自學順序）](#-學習路徑自學順序)
- [開發環境設置](#-開發環境設置)
- [專案結構](#-專案結構)
- [範例說明](#-範例說明)
- [常見問題（疑難排解）](#-常見問題疑難排解)
- [參考：開發環境版本](#-參考開發環境版本)
- [版本歷史](#-版本歷史)

## 🚀 快速開始（建議先做）

### 1) 設定環境並安裝相依套件

依照下方「開發環境設置」完成環境建立與相依套件安裝。

### 2) 先跑環境檢測，確認一切正常

```powershell
python 01-test.py
```

看到 Python / MediaPipe / OpenCV 版本輸出，即表示基礎環境可用。

### 3) 選一個你最想看的展示直接跑

- 單張圖片物件偵測：先準備 `image.jpg`，再執行

```powershell
python 02-objectDetect.py
```

- USB 攝影機串流物件偵測

```powershell
python 03-objectDetect_stream.py
```

- USB 攝影機串流手部地標

```powershell
python 04-hand_landmark.py
```

## 🧭 學習路徑（自學順序）

建議依序學習（由簡到難）：

1. [01-test.py](01-test.py)：確認 Python / MediaPipe / OpenCV 與 GPU 等基礎狀態
2. [02-objectDetect.py](02-objectDetect.py)：理解 Task API 的「載入模型 → 推論 → 讀取結果」流程
3. [03-objectDetect_stream.py](03-objectDetect_stream.py)：把推論搬到即時串流，熟悉 OpenCV 影像迴圈與疊圖
4. [04-hand_landmark.py](04-hand_landmark.py)：理解地標（landmark）與可視化，延伸到手勢或人機互動

## 🛠️ 開發環境設置

### 1) 建立並啟用 Conda 環境

```powershell
# 建立環境（Python 3.12）
conda create -n mp-py312 python=3.12 -y

# 啟用環境
conda activate mp-py312

# 更新 pip
python -m pip install --upgrade pip
```

### 2) 安裝相依套件

註：OpenCV 建議「擇一」安裝；PyTorch 為可選（主要用於環境檢測時顯示 CUDA 相關資訊）。

```powershell
# 安裝 MediaPipe
pip install mediapipe

# 安裝 OpenCV（擇一）
pip install opencv-python

# 或：安裝 OpenCV（完整版本，包含額外貢獻模組）
# pip install opencv-contrib-python

# 安裝 PyTorch（請參照官方安裝指南選擇適合版本）
# 官方安裝指南：https://pytorch.org/get-started/locally/
# 範例（OS:windows package:pip language:python computePlatform:CUDA 13.0 版本）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安裝系統資訊工具（可選）
pip install psutil
```

## 📁 專案結構

```
mp-py312/
├── 01-test.py                 # 環境檢測：Python/MediaPipe/OpenCV/GPU/CUDA
├── 02-objectDetect.py         # 物件偵測（單張圖片，MediaPipe Tasks）
├── 03-objectDetect_stream.py  # 物件偵測（USB 攝影機串流，MediaPipe Tasks + OpenCV）
├── 04-hand_landmark.py        # 手部地標（USB 攝影機串流，左手，HandLandmarker + OpenCV）
├── efficientdet_lite0.tflite  # 物件偵測模型（可能由程式第一次執行時下載）
├── hand_landmarker.task       # 手部地標模型（可能由程式第一次執行時下載）
└── README.md                  # 專案說明文件
```

## 🧩 範例說明

### 01｜環境檢測

- 腳本：[01-test.py](01-test.py)
- 用途：輸出 Python / MediaPipe / OpenCV 版本，並檢查 GPU、CUDA、cuDNN 等資訊（若有安裝相應套件）。
- 執行：

```powershell
python 01-test.py
```

用途（教學建議）：先用這支確認環境，遇到問題也用它回報版本。

### 02｜物件偵測（單張圖片）

- 腳本：[02-objectDetect.py](02-objectDetect.py)
- 用途：讀取 `image.jpg`，使用 MediaPipe 官方 EfficientDet Lite0 模型，於終端輸出偵測框、標籤與分數。
- 需求：
	- 專案根目錄需放置 `image.jpg`
	- 第一次執行會自動下載模型 `efficientdet_lite0.tflite`（需網路）
	- 模型來源：MediaPipe 官方提供之 EfficientDet Lite0（COCO 2017 資料集，90 類別）

執行：

```powershell
python 02-objectDetect.py
```

<details>
<summary>可偵測類別（COCO 90 類，英／中對照，點此展開）</summary>

| English           | 正體中文   | English            | 正體中文   | English           | 正體中文   |
|-------------------|------------|--------------------|------------|-------------------|------------|
| person            | 人物       | bicycle            | 腳踏車     | car               | 汽車       |
| motorcycle        | 機車       | airplane           | 飛機       | bus               | 公車       |
| train             | 火車       | truck              | 卡車       | boat              | 船舶       |
| traffic light     | 交通號誌   | fire hydrant       | 消防栓     | stop sign         | 停止標誌   |
| parking meter     | 停車計時器 | bench              | 長椅       | bird              | 鳥         |
| cat               | 貓         | dog                | 狗         | horse             | 馬         |
| sheep             | 綿羊       | cow                | 牛         | elephant          | 大象       |
| bear              | 熊         | zebra              | 斑馬       | giraffe           | 長頸鹿     |
| backpack          | 背包       | umbrella           | 雨傘       | handbag           | 手提包     |
| tie               | 領帶       | suitcase           | 行李箱     | frisbee           | 飛盤       |
| skis              | 滑雪板     | snowboard          | 滑雪板（雪板） | sports ball       | 球類       |
| kite              | 風箏       | baseball bat       | 棒球棒     | baseball glove    | 棒球手套   |
| skateboard        | 滑板       | surfboard          | 衝浪板     | tennis racket     | 網球拍     |
| bottle            | 瓶子       | wine glass         | 酒杯       | cup               | 杯子       |
| fork              | 叉子       | knife              | 刀         | spoon             | 湯匙       |
| bowl              | 碗         | banana             | 香蕉       | apple             | 蘋果       |
| sandwich          | 三明治     | orange             | 柳橙       | broccoli          | 青花菜     |
| carrot            | 紅蘿蔔     | hot dog            | 熱狗       | pizza             | 披薩       |
| donut             | 甜甜圈     | cake               | 蛋糕       | chair             | 椅子       |
| couch             | 沙發       | potted plant       | 盆栽       | bed               | 床         |
| dining table      | 餐桌       | toilet             | 馬桶       | tv                | 電視       |
| laptop            | 筆電       | mouse              | 滑鼠       | remote            | 遙控器     |
| keyboard          | 鍵盤       | cell phone         | 手機       | microwave         | 微波爐     |
| oven              | 烤箱       | toaster            | 烤麵包機   | sink              | 水槽       |
| refrigerator      | 冰箱       | book               | 書本       | clock             | 時鐘       |
| vase              | 花瓶       | scissors           | 剪刀       | teddy bear        | 泰迪熊     |
| hair drier        | 吹風機     | toothbrush         | 牙刷       | -                 | -          |

</details>

註：若模型下載失敗或 `image.jpg` 不存在，腳本會拋出錯誤並停止；請先確認網路與檔案路徑。

### 03｜物件偵測（USB 攝影機串流）

- 腳本：[03-objectDetect_stream.py](03-objectDetect_stream.py)
- 用途：使用 USB 攝影機即時偵測物件，於視窗顯示即時影像，並在畫面上標示物件名稱（含分數）。
- 需求：
	- 需可正常開啟攝影機（預設攝影機 ID = 0）
	- 第一次執行會自動下載模型 `efficientdet_lite0.tflite`（需網路）

執行：

```powershell
python 03-objectDetect_stream.py
```

操作方式：

- 會開啟視窗顯示即時影像與偵測框
- 按 `q` 鍵離開

若無法開啟攝影機，請在程式內調整 `DEFAULT_CAMERA_ID`（例如改成 1、2）後重試。

### 04｜手部地標（USB 攝影機串流）

- 腳本：[04-hand_landmark.py](04-hand_landmark.py)
- 用途：使用 USB 攝影機即時偵測左手地標（21 個關鍵點：0~20），於視窗顯示即時影像，並在每個地標點旁標示編號。
- 需求：
	- 需可正常開啟攝影機（預設攝影機 ID = 0）
	- 第一次執行會自動下載模型 `hand_landmarker.task`（需網路）
	- 模型來源：MediaPipe 官方提供之 HandLandmarker（float16 版本）

<img width="908" height="548" alt="image" src="https://github.com/user-attachments/assets/083c3b52-ff25-4986-89f7-c1cbb05341c8" />

執行：

```powershell
python 04-hand_landmark.py
```

操作方式：

- 會開啟視窗顯示即時影像與左手地標點（紅色圓點 + 編號）
- 按 `q` 鍵離開

手部地標點（0~20）摘要：

- **0**：WRIST（手腕）
- **1-4**：THUMB（拇指）：CMC / MCP / IP / TIP
- **5-8**：INDEX_FINGER（食指）：MCP / PIP / DIP / TIP
- **9-12**：MIDDLE_FINGER（中指）：MCP / PIP / DIP / TIP
- **13-16**：RING_FINGER（無名指）：MCP / PIP / DIP / TIP
- **17-20**：PINKY（小指）：MCP / PIP / DIP / TIP

備註：本範例僅顯示「左手」地標；若需偵測右手或雙手，可修改程式內的篩選邏輯。

## ❓ 常見問題（疑難排解）

- 物件偵測單張圖片找不到 `image.jpg`
	- 請將圖片命名為 `image.jpg` 並放在專案根目錄（與腳本同層）
- 第一次執行下載模型失敗
	- 請確認網路可連線；或先手動下載模型檔案放到專案根目錄
- 攝影機無法開啟（黑畫面或直接結束）
	- 先確認其他軟體沒有佔用攝影機
	- 於程式內調整 `DEFAULT_CAMERA_ID`（0 → 1 → 2）
- 視窗可以開但按鍵沒反應
	- 請確認焦點在 OpenCV 視窗上，再按 `q` 離開

## 🧾 參考：開發環境版本

以下摘要為執行 [01-test.py](01-test.py) 取得（更新：2025年12月30日）：

- **作業系統**: Windows 11 (10.0.26100)
- **Python**: 3.12.12（Anaconda）
- **MediaPipe**: 0.10.31
- **OpenCV**: 4.12.0（CUDA：未啟用）
- **PyTorch**: 2.9.1+cu130（CUDA 13.0 / cuDNN 91200）
- **GPU**: NVIDIA GeForce RTX 3060（12GB）

## 📝 版本歷史

- **2025-12-30**: 初始化專案，建立環境測試程式
- **2025-12-30**: 新增 02-objectDetect (MediaPipe Tasks) 單張圖片物件偵測範例，並自動下載 EfficientDet Lite0 模型
- **2025-12-30**: 新增 03-objectDetect_stream (MediaPipe Tasks + OpenCV) USB 攝影機串流物件偵測範例
- **2025-12-31**: 新增 04-hand_landmark (MediaPipe HandLandmarker) USB 攝影機左手地標偵測範例（21 點）
