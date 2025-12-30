"""
使用 MediaPipe Tasks 進行 USB 攝影機即時物件偵測，結果顯示於視窗。

架構說明：
- DetectorConfig: 偵測器設定（dataclass），集中管理可調參數
- DetectionResult: 單筆偵測結果（dataclass），方便後續處理與擴充
- StreamObjectDetector: 串流物件偵測器類別，封裝模型載入、偵測、視覺化
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

if TYPE_CHECKING:
    from mediapipe.tasks.vision import ObjectDetector as MediaPipeObjectDetector  # type: ignore[import-untyped]

# ==============================================================================
# 常數設定
# ==============================================================================

DEFAULT_MODEL_PATH = "efficientdet_lite0.tflite"
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)
DEFAULT_CAMERA_ID = 0
WINDOW_NAME = "MediaPipe 物件偵測 (按 'q' 離開)"


# ==============================================================================
# 資料類別
# ==============================================================================


@dataclass
class DetectorConfig:
    """物件偵測器設定，集中管理可調參數。"""

    model_path: str = DEFAULT_MODEL_PATH
    model_url: str = DEFAULT_MODEL_URL
    score_threshold: float = 0.5
    max_results: int = 5
    camera_id: int = DEFAULT_CAMERA_ID


@dataclass
class DetectionResult:
    """單筆偵測結果，方便後續處理與擴充。"""

    label: str
    score: float
    x: int
    y: int
    width: int
    height: int

    def get_bbox(self) -> tuple[int, int, int, int]:
        """回傳邊界框座標 (x1, y1, x2, y2)。"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


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
# 串流物件偵測器類別
# ==============================================================================


class StreamObjectDetector:
    """封裝 MediaPipe 串流物件偵測功能，便於重複使用與擴充。"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """初始化偵測器，載入模型。"""
        self.config = config or DetectorConfig()
        self._detector = self._create_detector()
        self._cap: cv2.VideoCapture | None = None

    def _create_detector(self) -> "MediaPipeObjectDetector":
        """建立 MediaPipe 物件偵測器（VIDEO 模式）。"""
        ensure_file_exists(self.config.model_path, self.config.model_url)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.config.model_path)
        )
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=self.config.score_threshold,
            max_results=self.config.max_results,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        return mp.tasks.vision.ObjectDetector.create_from_options(options)

    def _parse_detections(self, detections: list[Any]) -> list[DetectionResult]:
        """將 MediaPipe Detection 清單轉換為 DetectionResult 清單。"""
        results = []
        for detection in detections:
            box = detection.bounding_box
            x = max(0, box.origin_x)
            y = max(0, box.origin_y)
            width = box.width
            height = box.height

            if detection.categories:
                top = detection.categories[0]
                label = top.category_name or "(no label)"
                score = top.score
            else:
                label = "(no label)"
                score = 0.0

            results.append(
                DetectionResult(
                    label=label, score=score, x=x, y=y, width=width, height=height
                )
            )
        return results

    def _draw_detections(
        self, frame: np.ndarray, detections: list[DetectionResult]
    ) -> np.ndarray:
        """在影像上繪製偵測框與標籤。"""
        for det in detections:
            x1, y1, x2, y2 = det.get_bbox()

            # 繪製邊界框（綠色）
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 準備標籤文字
            label_text = f"{det.label} {det.score:.2f}"

            # 計算文字背景區域
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # 繪製文字背景（綠色填滿）
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                (0, 255, 0),
                -1,
            )

            # 繪製標籤文字（黑色）
            cv2.putText(
                frame,
                label_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return frame

    def _open_camera(self) -> cv2.VideoCapture:
        """開啟攝影機。"""
        cap = cv2.VideoCapture(self.config.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機（ID: {self.config.camera_id}）")
        return cap

    def run(self) -> None:
        """執行即時物件偵測，顯示於視窗。"""
        self._cap = self._open_camera()
        frame_count = 0

        print(f"攝影機已開啟（ID: {self.config.camera_id}）")
        print("按 'q' 鍵退出")

        try:
            while True:
                ret, frame = self._cap.read() # 讀取攝影機畫面, ret表示是否成功
                if not ret:
                    print("無法讀取攝影機畫面")
                    break

                # 轉換為 MediaPipe Image 格式（RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) # 建立 MediaPipe Image

                # 執行物件偵測（VIDEO 模式需要時間戳記）
                frame_count += 1 # 計數幀數
                timestamp_ms = int(frame_count * (1000 / 30))  # 假設 30 FPS
                result = self._detector.detect_for_video(mp_image, timestamp_ms)

                # 解析偵測結果
                detections = self._parse_detections(result.detections or []) # 轉換為 DetectionResult 清單

                # 繪製偵測結果
                frame = self._draw_detections(frame, detections)

                # 顯示偵測數量資訊
                info_text = f"Detections: {len(detections)}"
                cv2.putText(
                    frame,
                    info_text,
                    (10, 30), # 位置
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, # 字型大小
                    (0, 255, 0), # 顏色 (綠色)
                    2, # 粗細
                )

                # 顯示影像
                cv2.imshow(WINDOW_NAME, frame)

                # 按 'q' 離開
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("使用者中斷程式")
                    break

        finally:
            if self._cap: # 確保攝影機資源釋放
                self._cap.release()
            cv2.destroyAllWindows()
            print("攝影機已關閉")


# ==============================================================================
# 主程式
# ==============================================================================


def main() -> None:
    """主程式入口。"""
    config = DetectorConfig()
    detector = StreamObjectDetector(config)
    detector.run()


if __name__ == "__main__":
    main()
