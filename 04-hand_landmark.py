""" 
使用 MediaPipe Tasks (HandLandmarker) + OpenCV 進行 USB 攝影機即時手部地標偵測（左手）。

功能說明：
- 使用 USB Webcam 擷取即時影像
- 偵測手部地標（MediaPipe Hands：官方編號 0~20）
- 僅顯示「左手」的地標點位，並在每個點旁標示編號

架構說明（依循 02-objectDetect.py 的風格）：
- HandLandmarkConfig: 偵測器設定（dataclass），集中管理可調參數
- StreamHandLandmarker: 串流手部地標偵測器類別，封裝攝影機、推論、視覺化

操作方式：
- 執行後會開啟視窗顯示即時影像
- 按下 'q' 離開

備註：
- MediaPipe Hands 的 21 個地標點：0 是手腕（wrist），1~20 為手指關節點（共 20 個）
"""

from __future__ import annotations # 允許類別內部使用類別本身作為型別註解

from dataclasses import dataclass # 提供簡易的資料類別定義
from pathlib import Path # 用於處理檔案路徑
from typing import Any # 用於型別註解, Any 表示任意型別,typing 模組提供多種型別註解工具
from urllib.request import urlretrieve # 用於從 URL 下載檔案

import cv2
import mediapipe as mp
import numpy as np


# ==============================================================================
# 常數設定
# ==============================================================================

DEFAULT_CAMERA_ID = 0 # 預設webcam ID
WINDOW_NAME = "MediaPipe Left_Hand_landmarker (press 'q' Exit)" # 視窗名稱
DEFAULT_MODEL_PATH = "hand_landmarker.task" # 手部地標模型檔案路徑
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_FPS = 30  # 估計攝影機 FPS，供 timestamp 使用


# ==============================================================================
# 工具函式
# ==============================================================================


def ensure_file_exists(path: str | Path, download_url: str | None = None) -> Path:
    """確保檔案存在，必要時從指定網址下載。"""
    file_path = Path(path)
    if file_path.exists():
        return file_path

    if not download_url:
        raise FileNotFoundError(f"找不到必要檔案: {file_path}")

    print(f"正在下載模型至 {file_path} ...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(download_url, file_path) # 從指定網址下載檔案,urlretrieve 函式會將檔案下載到指定路徑
    print("模型下載完成")
    return file_path


# ==============================================================================
# 資料類別
# ==============================================================================


@dataclass
class HandLandmarkConfig:
    """手部地標偵測設定，集中管理可調參數。"""

    camera_id: int = DEFAULT_CAMERA_ID # 預設攝影機 ID
    model_path: str = DEFAULT_MODEL_PATH 
    model_url: str = DEFAULT_MODEL_URL
    target_fps: int = DEFAULT_FPS  # 供 timestamp 計算

    # MediaPipe HandLandmarker 參數
    num_hands: int = 2 # 偵測手數上限
    min_hand_detection_confidence: float = 0.5 # 最小手部偵測信心值 , 用於初始手部偵測, 較高值可提升準確度
    min_hand_presence_confidence: float = 0.5 # 最小手部存在信心值, 用於手部偵測後續步驟, 較高值可提升準確度
    min_tracking_confidence: float = 0.5 # 最小追蹤信心值, 用於手部追蹤, 較高值可提升穩定性

    # 關節數字視覺化參數
    draw_wrist: bool = True  # 是否繪製手腕（編號 0）, True 表示繪製，False 表示只繪製 1~20（20 個關節點）
    point_radius: int = 4 # 關節圓點半徑
    point_color_bgr: tuple[int, int, int] = (0, 255, 0)  # 關節點顏色 (BGR 格式)
    point_thickness: int = -1  # -1 表示填滿
    text_scale: float = 0.4 # 文字縮放比例
    text_thickness: int = 1 # 字體粗細


# ==============================================================================
# 串流手部地標偵測器類別
# ==============================================================================


class StreamHandLandmarker:
    """封裝 MediaPipe Tasks HandLandmarker 串流偵測（左手），便於重複使用與後續擴充。"""

    def __init__(self, config: HandLandmarkConfig | None = None) -> None: # 初始化偵測器, 可傳入自訂設定, 否則使用預設設定
        self.config = config or HandLandmarkConfig() # 使用預設設定, 若無提供自訂設定
        self._cap: cv2.VideoCapture | None = None # 攝影機物件, 初始為 None, 待開啟攝影機後賦值, 方便後續釋放資源
        self._landmarker = self._create_landmarker() # 建立 MediaPipe HandLandmarker 物件 ,供後續偵測使用, 透過私有方法建立, 封裝細節, 提升可讀性

    def _create_landmarker(self) -> Any: # 建立 MediaPipe HandLandmarker 物件 , 供後續偵測使用
        """建立 MediaPipe HandLandmarker（VIDEO 模式）。"""
        ensure_file_exists(self.config.model_path, self.config.model_url) # 確保模型檔案存在, 若不存在則下載

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.config.model_path) # 模型檔案路徑
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options, # 設定基本選項, 包含模型路徑
            num_hands=self.config.num_hands, # 偵測手數上限
            min_hand_detection_confidence=self.config.min_hand_detection_confidence, # 最小手部偵測信心值
            min_hand_presence_confidence=self.config.min_hand_presence_confidence, # 最小手部存在信心值
            min_tracking_confidence=self.config.min_tracking_confidence, # 最小追蹤信心值
            running_mode=mp.tasks.vision.RunningMode.VIDEO, # 設定為 VIDEO 模式, 因為是串流影像, 需要時間戳記, 以提升追蹤效果
        ) # 設定 HandLandmarker 參數
        return mp.tasks.vision.HandLandmarker.create_from_options(options) # 建立 HandLandmarker 物件並回傳, 供後續偵測使用

    def _open_camera(self) -> cv2.VideoCapture:
        """開啟攝影機。"""
        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機（ID: {self.config.camera_id}）")
        return cap

    @staticmethod # 判斷是否為左手, 使用 MediaPipe 的 handedness 分類結果, 回傳布林值, True 表示左手, False 表示非左手,回傳到呼叫處理
    # 這個@staticmethod 裝飾器表示此方法不依賴於類別實例的狀態, 可直接透過類別名稱呼叫
    def _is_left_hand(category: Any) -> bool: 
        """判斷此手是否為左手（以 MediaPipe 的 handedness 分類為準）。"""
        try:
            label = getattr(category, "category_name", "") 
            # 大小寫不敏感比對，避免未來版本差異
            return label.lower() == "left" # 回傳是否為左手, True 表示左手, False 表示非左手
        except Exception:
            return False

    def _select_left_hand_landmarks(self, result: Any) -> Any | None: # Any 為 MediaPipe HandLandmarker 的推論結果類型
        """從推論結果中挑出左手的 landmarks（若不存在則回傳 None）。"""
        if not result or not result.hand_landmarks or not result.handedness:
            return None

        # handedness[i] 對應 hand_landmarks[i]
        for i, lm_list in enumerate(result.hand_landmarks): 
            # 遍歷所有偵測到的手部地標, 透過 enumerate 同時取得索引與地標列表, lm_list 為第 i 隻手的地標列表
            if i < len(result.handedness): # 確保索引不超出範圍, 範圍為 handedness 清單長度,handedness 清單包含每隻手的分類結果
                # handedness[i] 是 Classification 列表，取第一個
                if result.handedness[i] and self._is_left_hand(result.handedness[i][0]):
                    # 判斷是否為左手, 使用私有方法 _is_left_hand, 傳入 handedness[i][0] 作為分類結果, 回傳布林值
                    # handedness[i][0] 為第一個分類結果, 因為 MediaPipe 可能會回傳多個分類結果, 但通常只取第一個
                    # handedness[i][0] 的型別為 Classification,內容為 category_name("Left" 或 "Right") 與 score (信心值)
                    return lm_list # 回傳左手的地標列表

        return None # 若無左手則回傳 None

    def _draw_landmark_index(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        index: int,
    ) -> None:
        """在指定座標繪製地標點與編號。"""

        # 1) 繪製點
        cv2.circle(
            frame,
            (x, y),
            self.config.point_radius, # 點半徑
            self.config.point_color_bgr,
            self.config.point_thickness,
        )

        # 2) 繪製編號（綠色文字，無背景）
        text = str(index) # index為地標編號,由呼叫處傳入

        # 文字位置（稍微偏移，避免蓋到點）
        tx = x + self.config.point_radius + 2
        ty = y - self.config.point_radius - 2

        cv2.putText(
            frame,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale,
            self.config.point_color_bgr,  # 使用綠色（與點相同）
            self.config.text_thickness,
        )

    @staticmethod # 在影像左上角顯示提示文字(這裡的@staticmethod 裝飾器表示此方法不依賴於類別實例的狀態, 可直接透過類別名稱呼叫)
    def _draw_info_text(frame: np.ndarray, text: str) -> None:
        """在左上角顯示提示文字。"""
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    def _draw_left_hand_landmarks(self, frame: np.ndarray, landmarks: list[Any]) -> None:
        """將左手 landmarks 依照官方編號繪製到影像上。"""
        height, width = frame.shape[:2]

        # landmarks 為 21 個點（0~20）
        for idx, lm in enumerate(landmarks):
            # 若不顯示 wrist，則跳過 0
            if idx == 0 and not self.config.draw_wrist:
                continue

            # MediaPipe 提供的是正規化座標（0~1），需換算回像素座標
            x = int(lm.x * width)
            y = int(lm.y * height)

            # 邊界保護（避免座標落在畫面外）
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))

            self._draw_landmark_index(frame, x, y, idx)

    def run(self) -> None: 
        # run 方法為主程式入口, 執行即時左手地標偵測並顯示於視窗
        # self參數代表類別實例本身, 可透過 self 訪問類別屬性與方法, 包含的配置參數總共有 camera_id, model_path, model_url, target_fps 等等
        # -> None: 表示此方法無回傳值
        """執行即時左手地標偵測，顯示於視窗。"""
        self._cap = self._open_camera() # 開啟攝影機, 並賦值給類別屬性 _cap, 方便後續釋放資源
        frame_count = 0 # 記錄影格數, 用於計算 timestamp

        print(f"攝影機已開啟（ID: {self.config.camera_id}）")
        print("僅顯示左手地標（官方編號 0~20）")
        print("按 'q' 鍵退出")

        try:
            while True:
                ret, frame = self._cap.read()  # 讀取攝影機畫面, ret 表示是否成功
                if not ret:
                    print("無法讀取攝影機畫面")
                    break

                # 轉換為 RGB（MediaPipe 需要 RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # 執行手部地標偵測（VIDEO 模式需要時間戳記）
                frame_count += 1
                timestamp_ms = int(frame_count * (1000 / self.config.target_fps))
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

                # 只取左手
                left_hand_lm = self._select_left_hand_landmarks(result)
                if left_hand_lm is not None:
                    self._draw_left_hand_landmarks(frame, left_hand_lm)

                # 顯示簡易資訊
                info_text = "Left hand: DETECTED" if left_hand_lm is not None else "Left hand: NONE"
                self._draw_info_text(frame, info_text)

                cv2.imshow(WINDOW_NAME, frame)

                # 按 'q' 離開
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("使用者中斷程式")
                    break

        finally:
            if self._cap:
                self._cap.release()  # 確保攝影機資源釋放
            cv2.destroyAllWindows()
            print("攝影機已關閉")


# ==============================================================================
# 主程式
# ==============================================================================


def main() -> None:
    """主程式入口。"""
    config = HandLandmarkConfig() # 使用預設設定, HandLandmarkConfig 為前面建立的資料類別, 可集中管理可調參數
    landmarker = StreamHandLandmarker(config) # 建立串流手部地標偵測器
    landmarker.run() # 執行即時左手地標偵測


if __name__ == "__main__":
    main()

