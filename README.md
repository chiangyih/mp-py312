# MediaPipe Python 3.12 專案

> **最後更新**: 2025年12月30日

這是一個基於 Python 3.12 的 MediaPipe 開發環境專案。

## 📁 專案結構

```
mp-py312/
├── 01-test.py              # 環境測試程式
├── 02-objectDetect.py      # 單張圖片物件偵測 (MediaPipe Tasks)
└── README.md               # 專案說明文件
```

## 🛠️ 環境資訊

以下是本專案的完整環境配置資訊（最後更新：2025年12月30日）：

### Python 環境

- **Python 版本**: 3.12.12
- **Python 編譯器**: MSC v.1929 64 bit (AMD64)
- **Python 發行版**: Anaconda, Inc.
- **作業系統**: Windows 11 (10.0.26100)
- **系統架構**: AMD64
- **處理器**: Intel64 Family 6 Model 94 Stepping 3, GenuineIntel

### 已安裝套件

#### MediaPipe
- **版本**: 0.10.31
- **安裝路徑**: `C:\Users\tseng\miniconda3\envs\mp-py312\Lib\site-packages\mediapipe\`

#### OpenCV
- **版本**: 4.12.0
- **安裝路徑**: `C:\Users\tseng\miniconda3\envs\mp-py312\Lib\site-packages\cv2\`
- **CUDA 支援**: ❌ 未啟用

#### PyTorch
- **版本**: 2.9.1+cu130
- **CUDA 版本**: 13.0
- **cuDNN 版本**: 91200
- **CUDA 可用**: ✅ 是

### 🎮 GPU 資訊

本環境支援 GPU 加速運算：

- **GPU 裝置數量**: 1
- **GPU 型號**: NVIDIA GeForce RTX 3060
- **顯示記憶體**: 12.00 GB
- **CUDA 核心數**: 28 個多處理器
- **計算能力**: 8.6

### ⚠️ 注意事項

- OpenCV 目前使用 CPU 版本，未啟用 CUDA 支援
- TensorFlow 未安裝
- psutil 未安裝（無法顯示詳細系統記憶體資訊）

## 🚀 快速開始

### 啟用 Conda 環境

```powershell
conda activate mp-py312
```

### 執行環境測試

```powershell
python 01-test.py
```

此程式會自動檢測並顯示：
- Python 版本資訊
- MediaPipe 版本
- OpenCV 版本與 CUDA 支援
- GPU 與 CUDA 資訊（PyTorch、TensorFlow）
- 系統資訊

## 🎯 物件偵測範例 (MediaPipe Tasks)

- 範例腳本：[02-objectDetect.py](02-objectDetect.py)
- 功能：讀取 `image.jpg`，使用 MediaPipe Tasks 物件偵測模型（EfficientDet Lite0），於終端輸出偵測框、標籤與分數。
- 需求：
	- 專案根目錄需放置 `image.jpg`
	- 第一次執行會自動下載模型 `efficientdet_lite0.tflite`（需網路）
	- 模型來源：MediaPipe 官方提供之 EfficientDet Lite0 物件偵測模型（COCO 2017 資料集，90 類別）
	- 可偵測類別（COCO 90 類，英／中對照）：

| English              | 正體中文             | English                 | 正體中文          |
|----------------------|----------------------|-------------------------|-------------------|
| person               | 人物                 | backpack                | 背包              |
| bicycle              | 腳踏車               | umbrella                | 雨傘              |
| car                  | 汽車                 | handbag                 | 手提包            |
| motorcycle           | 機車                 | tie                     | 領帶              |
| airplane             | 飛機                 | suitcase                | 行李箱            |
| bus                  | 公車                 | frisbee                 | 飛盤              |
| train                | 火車                 | skis                    | 滑雪板            |
| truck                | 卡車                 | snowboard               | 滑雪板（雪板）    |
| boat                 | 船舶                 | sports ball             | 球類              |
| traffic light        | 交通號誌             | kite                    | 風箏              |
| fire hydrant         | 消防栓               | baseball bat            | 棒球棒            |
| stop sign            | 停止標誌             | baseball glove          | 棒球手套          |
| parking meter        | 停車計時器           | skateboard              | 滑板              |
| bench                | 長椅                 | surfboard               | 衝浪板            |
| bird                 | 鳥                   | tennis racket           | 網球拍            |
| cat                  | 貓                   | bottle                  | 瓶子              |
| dog                  | 狗                   | wine glass              | 酒杯              |
| horse                | 馬                   | cup                     | 杯子              |
| sheep                | 綿羊                 | fork                    | 叉子              |
| cow                  | 牛                   | knife                   | 刀                |
| elephant             | 大象                 | spoon                   | 湯匙              |
| bear                 | 熊                   | bowl                    | 碗                |
| zebra                | 斑馬                 | banana                  | 香蕉              |
| giraffe              | 長頸鹿               | apple                   | 蘋果              |
| backpack             | 背包                 | sandwich                | 三明治            |
| umbrella             | 雨傘                 | orange                  | 柳橙              |
| handbag              | 手提包               | broccoli                | 青花菜            |
| tie                  | 領帶                 | carrot                  | 紅蘿蔔            |
| suitcase             | 行李箱               | hot dog                 | 熱狗              |
| frisbee              | 飛盤                 | pizza                   | 披薩              |
| skis                 | 滑雪板               | donut                   | 甜甜圈            |
| snowboard            | 滑雪板（雪板）       | cake                    | 蛋糕              |
| sports ball          | 球類                 | chair                   | 椅子              |
| kite                 | 風箏                 | couch                   | 沙發              |
| baseball bat         | 棒球棒               | potted plant            | 盆栽              |
| baseball glove       | 棒球手套             | bed                     | 床                |
| skateboard           | 滑板                 | dining table            | 餐桌              |
| surfboard            | 衝浪板               | toilet                  | 馬桶              |
| tennis racket        | 網球拍               | tv                      | 電視              |
| bottle               | 瓶子                 | laptop                  | 筆電              |
| wine glass           | 酒杯                 | mouse                   | 滑鼠              |
| cup                  | 杯子                 | remote                  | 遙控器            |
| fork                 | 叉子                 | keyboard                | 鍵盤              |
| knife                | 刀                   | cell phone              | 手機              |
| spoon                | 湯匙                 | microwave               | 微波爐            |
| bowl                 | 碗                   | oven                    | 烤箱              |
| banana               | 香蕉                 | toaster                 | 烤麵包機          |
| apple                | 蘋果                 | sink                    | 水槽              |
| sandwich             | 三明治               | refrigerator            | 冰箱              |
| orange               | 柳橙                 | book                    | 書本              |
| broccoli             | 青花菜               | clock                   | 時鐘              |
| carrot               | 紅蘿蔔               | vase                    | 花瓶              |
| hot dog              | 熱狗                 | scissors                | 剪刀              |
| pizza                | 披薩                 | teddy bear              | 泰迪熊            |
| donut                | 甜甜圈               | hair drier              | 吹風機            |
| cake                 | 蛋糕                 | toothbrush              | 牙刷              |

### 執行步驟

```powershell
python 02-objectDetect.py
```

### 預期輸出（示意）

```
檔案: image.jpg
偵測數量: 1
偵測 1: label=dog, score=0.86, box=(x=21, y=4, w=572, h=440)
```

若模型下載失敗或 `image.jpg` 不存在，腳本會拋出錯誤並停止；請確認網路與檔案路徑後重試。

## 📦 套件安裝

如需安裝額外套件：

```powershell
# 安裝 MediaPipe
pip install mediapipe

# 安裝 OpenCV
pip install opencv-python

# 安裝 PyTorch（CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# 安裝系統資訊工具（可選）
pip install psutil
```

## 💡 使用說明

本專案主要用於 MediaPipe 的開發與測試。MediaPipe 是 Google 開發的跨平台機器學習框架，支援多種視覺與音訊處理任務。

### 主要功能
- ✅ 人臉偵測
- ✅ 姿態估計
- ✅ 手部追蹤
- ✅ 物件偵測
- ✅ 影像分割

## 🔧 環境需求

- Python 3.12+
- Windows 11
- NVIDIA GPU（可選，用於 PyTorch 加速）
- CUDA 13.0+（可選）

## 📝 版本歷史

- **2025-12-30**: 初始化專案，建立環境測試程式
- **2025-12-30**: 新增 02-objectDetect (MediaPipe Tasks) 單張圖片物件偵測範例，並自動下載 EfficientDet Lite0 模型


---


