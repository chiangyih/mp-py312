"""
使用 MediaPipe Tasks 進行單張圖片的簡易物件偵測，結果輸出於終端機。

架構說明：
- DetectorConfig: 偵測器設定（dataclass），集中管理可調參數
- DetectionResult: 單筆偵測結果（dataclass），方便後續處理與擴充
- ObjectDetector: 物件偵測器類別，封裝模型載入、偵測、結果輸出
"""

from __future__ import annotations # 允許類別內部使用類別本身作為型別註解

from dataclasses import dataclass # 提供簡易的資料類別定義
from pathlib import Path # 用於處理檔案路徑
from typing import Any, TYPE_CHECKING    # 用於型別檢查時的條件匯入
from urllib.request import urlretrieve # 用於從 URL 下載檔案

import mediapipe as mp

if TYPE_CHECKING:
    from mediapipe.tasks.python.components.containers import Detection
    from mediapipe.tasks.vision import ObjectDetector as MediaPipeObjectDetector  # type: ignore[import-untyped]

# ==============================================================================
# 常數設定
# ==============================================================================

DEFAULT_IMAGE_PATH = "image.jpg"
DEFAULT_MODEL_PATH = "efficientdet_lite0.tflite"
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)


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


@dataclass
class DetectionResult:
    """單筆偵測結果，方便後續處理與擴充。"""

    index: int
    label: str
    score: float
    x: int
    y: int
    width: int
    height: int

    def __str__(self) -> str:  # 格式化輸出偵測結果
        return (
            f"偵測 {self.index}: label={self.label}, score={self.score:.2f}, "
            f"box=(x={self.x}, y={self.y}, w={self.width}, h={self.height})"
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


def validate_image(path: str | Path) -> Path:
    """檢查圖片是否存在。"""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到圖片檔案: {image_path}")
    return image_path


# ==============================================================================
# 物件偵測器類別
# ==============================================================================


class ObjectDetector:
    """封裝 MediaPipe 物件偵測功能，便於重複使用與擴充。"""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        """初始化偵測器，載入模型。"""
        self.config = config or DetectorConfig()
        self._detector = self._create_detector()

    def _create_detector(self) -> "MediaPipeObjectDetector":
        """建立 MediaPipe 物件偵測器。"""
        ensure_file_exists(self.config.model_path, self.config.model_url)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.config.model_path)
        )
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=self.config.score_threshold,
            max_results=self.config.max_results,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        return mp.tasks.vision.ObjectDetector.create_from_options(options)

    @staticmethod
    def _parse_detection(idx: int, detection: Any) -> DetectionResult:
        """將 MediaPipe Detection 轉換為 DetectionResult。"""
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

        return DetectionResult(
            index=idx, label=label, score=score, x=x, y=y, width=width, height=height
        )

    def detect(self, image_path: str | Path) -> list[DetectionResult]:
        """對指定圖片執行物件偵測，回傳結果清單。"""
        image_file = validate_image(image_path)
        mp_image = mp.Image.create_from_file(str(image_file))
        result = self._detector.detect(mp_image)

        return [
            self._parse_detection(idx, det)
            for idx, det in enumerate(result.detections or [], start=1)
        ]

    def detect_and_print(self, image_path: str | Path) -> list[DetectionResult]:
        """對指定圖片執行物件偵測並輸出結果至終端機。"""
        results = self.detect(image_path)

        print(f"檔案: {image_path}")
        print(f"偵測數量: {len(results)}")

        for r in results:
            print(r)

        return results


# ==============================================================================
# 主程式
# ==============================================================================


def main() -> None:
    """主程式入口。"""
    config = DetectorConfig()
    detector = ObjectDetector(config)
    detector.detect_and_print(DEFAULT_IMAGE_PATH)


if __name__ == "__main__":
    main()
