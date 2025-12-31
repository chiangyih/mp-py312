# MediaPipe 使用 Python 開發

> **最後更新**: 2025年12月31日

使用 MediaPipe Tasks 搭配 OpenCV 的多個範例腳本，涵蓋環境檢測、物件偵測（單張與串流）、手部地標偵測（串流）。

## 🧰 開發環境版本

以下摘要為執行 [01-test.py](01-test.py) 取得（更新：2025年12月30日）：

- **作業系統**: Windows 11 (10.0.26100)
- **Python**: 3.12.12（Anaconda）
- **MediaPipe**: 0.10.31
- **OpenCV**: 4.12.0（CUDA：未啟用）
- **PyTorch**: 2.9.1+cu130（CUDA 13.0 / cuDNN 91200）
- **GPU**: NVIDIA GeForce RTX 3060（12GB）

## 📁 專案結構

```
mp-py312/
├── 01-test.py                 # 環境檢測：Python/MediaPipe/OpenCV/GPU/CUDA
├── 02-objectDetect.py         # 物件偵測（單張圖片，MediaPipe Tasks）
├── 03-objectDetect_stream.py  # 物件偵測（USB 攝影機串流，MediaPipe Tasks + OpenCV）
├── 04-hand_landmark.py        # 手部地標（USB 攝影機串流，左手，HandLandmarker + OpenCV）
└── README.md                  # 專案說明文件
```

## 🧩 子程式分類與使用方式

### 01｜環境檢測

- 腳本：[01-test.py](01-test.py)
- 用途：輸出 Python / MediaPipe / OpenCV 版本，並檢查 GPU、CUDA、cuDNN 等資訊（若有安裝相應套件）。
- 執行：

```powershell
python 01-test.py
```

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

可偵測類別（COCO 90 類，英／中對照，6 欄）：

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

註：若模型下載失敗或 `image.jpg` 不存在，腳本會拋出錯誤並停止；請確認網路與檔案路徑後重試。

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

## 📦 套件安裝

如需安裝額外套件：

```powershell
# 安裝 MediaPipe
pip install mediapipe

# 安裝 OpenCV（基礎版本）
pip install opencv-python

# 安裝 OpenCV（完整版本，包含額外貢獻模組）
pip install opencv-contrib-python

# 安裝 PyTorch（請參照官方安裝指南選擇適合版本）
# 官方安裝指南：https://pytorch.org/get-started/locally/
# 範例（OS:windows package:pip language:python computePlatform:CUDA 13.0 版本）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安裝系統資訊工具（可選）
pip install psutil
```


## 📝 版本歷史

- **2025-12-30**: 初始化專案，建立環境測試程式
- **2025-12-30**: 新增 02-objectDetect (MediaPipe Tasks) 單張圖片物件偵測範例，並自動下載 EfficientDet Lite0 模型
- **2025-12-30**: 新增 03-objectDetect_stream (MediaPipe Tasks + OpenCV) USB 攝影機串流物件偵測範例
- **2025-12-31**: 新增 04-hand_landmark (MediaPipe HandLandmarker) USB 攝影機左手地標偵測範例（21 點）
