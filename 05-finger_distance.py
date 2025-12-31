""" 
使用 MediaPipe Tasks (HandLandmarker) + OpenCV 計算手指關節點距離。

功能說明：
- 使用 USB Webcam 擷取即時影像
- 偵測手部地標（MediaPipe Hands：官方編號 0~20）
- 僅顯示「左手」的地標點 4（拇指尖）和點 8（食指尖）
- 計算點 4 到點 8 的距離（單位：mm）
- 在畫面上繪製兩點之間的連線
- 在畫面上顯示距離數值

架構說明：
- FingerDistanceConfig: 偵測器設定（dataclass），集中管理可調參數
- FingerDistanceDetector: 手指距離偵測器類別，封裝攝影機、推論、距離計算、視覺化

操作方式：
- 執行後會開啟視窗顯示即時影像
- 按下 'q' 離開

備註：
- 關節點 4：THUMB_TIP（拇指尖）
- 關節點 8：INDEX_FINGER_TIP（食指尖）
- 距離計算使用假設：手掌寬度約為 85mm，用於像素轉換為實際距離
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

DEFAULT_CAMERA_ID = 0  # 預設 webcam ID
WINDOW_NAME = "Finger Distance (4-8) - Press 'q' to Exit"  # 視窗名稱
DEFAULT_MODEL_PATH = "hand_landmarker.task"  # 手部地標模型檔案路徑
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_FPS = 30  # 估計攝影機 FPS，供 timestamp 使用

# 距離計算參數
ASSUMED_HAND_WIDTH_MM = 100.0  # 假設手掌寬度（mm），用於像素轉實際距離的換算
REFERENCE_LANDMARK_INDICES = (0, 9)  # 用於估算手掌寬度的參考點（手腕到中指根部）

# 目標關節點
TARGET_LANDMARK_1 = 4  # 拇指尖
TARGET_LANDMARK_2 = 8  # 食指尖


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
class FingerDistanceConfig:
    """手指距離偵測設定，集中管理可調參數。"""

    camera_id: int = DEFAULT_CAMERA_ID
    model_path: str = DEFAULT_MODEL_PATH
    model_url: str = DEFAULT_MODEL_URL
    target_fps: int = DEFAULT_FPS

    # MediaPipe HandLandmarker 參數
    num_hands: int = 2  # 偵測手數上限
    min_hand_detection_confidence: float = 0.5  # 最小手部偵測信心值
    min_hand_presence_confidence: float = 0.5  # 最小手部存在信心值
    min_tracking_confidence: float = 0.5  # 最小追蹤信心值

    # 視覺化參數
    point_radius: int = 6  # 關節圓點半徑
    point_color_bgr: tuple[int, int, int] = (0, 255, 0)  # 關節點顏色（綠色）
    point_thickness: int = -1  # -1 表示填滿
    line_color_bgr: tuple[int, int, int] = (255, 0, 0)  # 連線顏色（藍色）
    line_thickness: int = 2  # 連線粗細
    text_scale: float = 0.5  # 文字縮放比例
    text_thickness: int = 2  # 字體粗細
    text_color_bgr: tuple[int, int, int] = (0, 255, 255)  # 文字顏色（黃色）

    # 距離計算參數
    assumed_hand_width_mm: float = ASSUMED_HAND_WIDTH_MM  # 假設手掌寬度（mm）


# ==============================================================================
# 手指距離偵測器類別
# ==============================================================================


class FingerDistanceDetector:
    """封裝 MediaPipe Tasks HandLandmarker 手指距離計算，便於重複使用與後續擴充。"""

    def __init__(self, config: FingerDistanceConfig | None = None) -> None:
        self.config = config or FingerDistanceConfig()
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
                if result.handedness[i] and self._is_left_hand(result.handedness[i][0]):
                    return lm_list

        return None

    def _calculate_pixel_distance(
        self,
        landmarks: list[Any],
        idx1: int,
        idx2: int,
        width: int,
        height: int,
    ) -> float:
        """計算兩個關節點之間的像素距離。"""
        # 取得兩個關節點的正規化座標（0~1）
        lm1 = landmarks[idx1]
        lm2 = landmarks[idx2]

        # 轉換為像素座標
        x1 = lm1.x * width
        y1 = lm1.y * height
        x2 = lm2.x * width
        y2 = lm2.y * height

        # 計算歐氏距離
        distance_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance_px

    def _estimate_pixel_to_mm_ratio(
        self,
        landmarks: list[Any],
        width: int,
        height: int,
    ) -> float:
        """估算像素到 mm 的轉換比例（基於手掌寬度）。"""
        # 使用關節點 0（手腕）到關節點 5（食指根部）的距離作為參考
        ref_distance_px = self._calculate_pixel_distance(
            landmarks, REFERENCE_LANDMARK_INDICES[0], REFERENCE_LANDMARK_INDICES[1], width, height
        )

        # 假設此距離約為手掌寬度的一半（經驗值）
        assumed_ref_distance_mm = self.config.assumed_hand_width_mm * 0.5

        # 計算轉換比例（mm/pixel）
        if ref_distance_px > 0:
            return assumed_ref_distance_mm / ref_distance_px
        return 1.0  # 避免除以零

    def _draw_landmark_with_label(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        label: str,
    ) -> None:
        """在指定座標繪製關節點與標籤。"""
        # 繪製圓點
        cv2.circle(
            frame,
            (x, y),
            self.config.point_radius,
            self.config.point_color_bgr,
            self.config.point_thickness,
        )

        # 繪製標籤文字（稍微偏移）
        tx = x + self.config.point_radius + 3
        ty = y - self.config.point_radius - 3

        cv2.putText(
            frame,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale,
            self.config.point_color_bgr,
            self.config.text_thickness,
        )

    def _draw_line_between_points(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """在兩個關節點之間繪製連線。"""
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            self.config.line_color_bgr,
            self.config.line_thickness,
        )

    def _draw_distance_text(
        self,
        frame: np.ndarray,
        distance_mm: float,
    ) -> None:
        """在畫面上顯示距離數值。"""
        text = f"Distance: {distance_mm:.1f} mm"

        # 顯示在畫面左上角
        cv2.putText(
            frame,
            text,
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale + 0.3,  # 稍大一點
            self.config.text_color_bgr,
            self.config.text_thickness,
        )

    def _process_and_draw(
        self,
        frame: np.ndarray,
        landmarks: list[Any],
    ) -> None:
        """處理左手地標，計算距離並繪製於畫面上。"""
        height, width = frame.shape[:2]

        # 取得關節點 4 和 8 的座標
        lm4 = landmarks[TARGET_LANDMARK_1]  # 拇指尖
        lm8 = landmarks[TARGET_LANDMARK_2]  # 食指尖

        # 轉換為像素座標
        x4 = int(lm4.x * width)
        y4 = int(lm4.y * height)
        x8 = int(lm8.x * width)
        y8 = int(lm8.y * height)

        # 邊界保護
        x4 = max(0, min(width - 1, x4))
        y4 = max(0, min(height - 1, y4))
        x8 = max(0, min(width - 1, x8))
        y8 = max(0, min(height - 1, y8))

        # 繪製關節點 4 和 8
        self._draw_landmark_with_label(frame, x4, y4, "4")
        self._draw_landmark_with_label(frame, x8, y8, "8")

        # 繪製連線
        self._draw_line_between_points(frame, x4, y4, x8, y8)

        # 計算像素距離
        distance_px = self._calculate_pixel_distance(
            landmarks, TARGET_LANDMARK_1, TARGET_LANDMARK_2, width, height
        )

        # 估算像素到 mm 的轉換比例
        px_to_mm_ratio = self._estimate_pixel_to_mm_ratio(landmarks, width, height)

        # 計算實際距離（mm）
        distance_mm = distance_px * px_to_mm_ratio

        # 顯示距離數值
        self._draw_distance_text(frame, distance_mm)

    def run(self) -> None:
        """執行即時手指距離偵測，顯示於視窗。"""
        self._cap = self._open_camera()
        frame_count = 0

        print(f"攝影機已開啟（ID: {self.config.camera_id}）")
        print("偵測左手，計算關節點 4（拇指尖）到關節點 8（食指尖）的距離")
        print("按 'q' 鍵退出")

        try:
            while True:
                ret, frame = self._cap.read()
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
                    self._process_and_draw(frame, left_hand_lm)

                # 顯示畫面
                cv2.imshow(WINDOW_NAME, frame)

                # 按 'q' 離開
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("使用者中斷程式")
                    break

        finally:
            if self._cap:
                self._cap.release()
            cv2.destroyAllWindows()
            print("攝影機已關閉")


# ==============================================================================
# 主程式
# ==============================================================================


def main() -> None:
    """主程式入口。"""
    config = FingerDistanceConfig()
    detector = FingerDistanceDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
