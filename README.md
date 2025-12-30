# MediaPipe Python 3.12 專案

> **最後更新**: 2025年12月30日

這是一個基於 Python 3.12 的 MediaPipe 開發環境專案。

## 📁 專案結構

```
mp-py312/
├── 01-test.py          # 環境測試程式
└── README.md           # 專案說明文件
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

## 📄 授權

本專案僅供學習與開發使用。

---

**注意**: 此 README 由環境測試程式自動產生的資訊更新。如需重新檢測環境，請執行 `python 01-test.py`。
