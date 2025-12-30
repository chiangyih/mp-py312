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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


# ==============================================================================
# 常數設定
# ==============================================================================

DEFAULT_CAMERA_ID = 0
WINDOW_NAME = "MediaPipe 左手地標 (按 'q' 離開)"
DEFAULT_MODEL_PATH = "hand_landmarker.task"
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


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
    urlretrieve(download_url, file_path)
    print("模型下載完成")
    return file_path


# ==============================================================================
# 資料類別
# ==============================================================================


@dataclass
class HandLandmarkConfig:
    """手部地標偵測設定，集中管理可調參數。"""

    camera_id: int = DEFAULT_CAMERA_ID
    model_path: str = DEFAULT_MODEL_PATH
    model_url: str = DEFAULT_MODEL_URL

    # MediaPipe HandLandmarker 參數
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # 視覺化參數
    draw_wrist: bool = True  # True：顯示 0 (wrist)；False：只顯示 1~20（20 個關節點）
    point_radius: int = 5
    point_color_bgr: tuple[int, int, int] = (0, 255, 0)  # 綠色
    point_thickness: int = -1  # -1 表示填滿
    text_scale: float = 0.6
    text_color_bgr: tuple[int, int, int] = (0, 0, 0)  # 黑色
    text_thickness: int = 2
    text_bg_color_bgr: tuple[int, int, int] = (0, 255, 0)  # 綠底
    text_padding: int = 2


# ==============================================================================
# 串流手部地標偵測器類別
# ==============================================================================


class StreamHandLandmarker:
    """封裝 MediaPipe Tasks HandLandmarker 串流偵測（左手），便於重複使用與後續擴充。"""

    def __init__(self, config: HandLandmarkConfig | None = None) -> None:
        self.config = config or HandLandmarkConfig()
        self._cap: cv2.VideoCapture | None = None
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self) -> Any:
        """建立 MediaPipe HandLandmarker（VIDEO 模式）。"""
        ensure_file_exists(self.config.model_path, self.config.model_url)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.config.model_path)
        )
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _open_camera(self) -> cv2.VideoCapture:
        """開啟攝影機。"""
        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機（ID: {self.config.camera_id}）")
        return cap

    @staticmethod
    def _is_left_hand(category: Any) -> bool:
        """判斷此手是否為左手（以 MediaPipe 的 handedness 分類為準）。"""
        try:
            label = getattr(category, "category_name", "")
            # 大小寫不敏感比對，避免未來版本差異
            return label.lower() == "left"
        except Exception:
            return False

    def _select_left_hand_landmarks(self, result: Any) -> Any | None:
        """從推論結果中挑出左手的 landmarks（若不存在則回傳 None）。"""
        if not result or not result.hand_landmarks or not result.handedness:
            return None

        # handedness[i] 對應 hand_landmarks[i]
        for i, lm_list in enumerate(result.hand_landmarks):
            if i < len(result.handedness):
                # handedness[i] 是 Classification 列表，取第一個
                if result.handedness[i] and self._is_left_hand(result.handedness[i][0]):
                    return lm_list

        return None

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
            self.config.point_radius,
            self.config.point_color_bgr,
            self.config.point_thickness,
        )

        # 2) 繪製編號（綠色文字，無背景）
        text = str(index)

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
        """執行即時左手地標偵測，顯示於視窗。"""
        self._cap = self._open_camera()
        frame_count = 0

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
                timestamp_ms = int(frame_count * (1000 / 30))  # 假設 30 FPS
                result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

                # 只取左手
                left_hand_lm = self._select_left_hand_landmarks(result)
                if left_hand_lm is not None:
                    self._draw_left_hand_landmarks(frame, left_hand_lm)

                # 顯示簡易資訊
                info_text = "Left hand: DETECTED" if left_hand_lm is not None else "Left hand: NONE"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

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
    config = HandLandmarkConfig()
    landmarker = StreamHandLandmarker(config)
    landmarker.run()


if __name__ == "__main__":
    main()

