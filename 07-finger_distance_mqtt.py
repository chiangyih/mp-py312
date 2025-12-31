""" 
使用 MediaPipe Tasks (HandLandmarker) + OpenCV 計算手指關節點距離（兩點校準版 + MQTT）。

功能說明：
- 使用 USB Webcam 擷取即時影像
- 偵測手部地標（MediaPipe Hands：官方編號 0~20）
- 僅顯示「左手」的地標點 4（拇指尖）和點 8（食指尖）
- 使用兩點校準法：
  1. 第一次校準：將關節點 4-8 的最小距離設為 0mm（按 'm' 鍵確認）
  2. 第二次校準：將關節點 4-8 的最大距離設為 130mm（按 'M' 或 Shift+m 鍵確認）
  3. 之後的測量都基於這兩個校準點進行線性計算
- 在畫面上繪製兩點之間的連線
- 在畫面上顯示距離數值
- 每 2 秒透過 MQTT 發送距離數值

架構說明：
- FingerDistanceConfig: 偵測器設定（dataclass），集中管理可調參數
- CalibrationState: 校準狀態列舉
- FingerDistanceDetectorCalibrated: 手指距離偵測器類別（含兩點校準 + MQTT）

操作方式：
- 執行後會開啟視窗顯示即時影像
- 第一步：將手指併攏（讓關節點 4 和 8 最接近），按 'm' 鍵設定為 0mm
- 第二步：將手指張開（讓關節點 4 和 8 最遠），按 'M'（Shift+m）鍵設定為 130mm
- 校準完成後，即可正常測量並自動發送到 MQTT broker
- 按下 'r' 可重新校準
- 按下 'q' 離開

備註：
- 關節點 4：THUMB_TIP（拇指尖）
- 關節點 8：INDEX_FINGER_TIP（食指尖）
- MQTT 發送間隔：每 2 秒發送一次
- MQTT 發送格式：純數值字串（例如：12.34）
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import paho.mqtt.client as mqtt


# ==============================================================================
# 常數設定
# ==============================================================================

DEFAULT_CAMERA_ID = 0  # 預設 webcam ID
WINDOW_NAME = "Finger Distance (Calibrated + MQTT) - 'm'=0mm, 'M'=130mm, 'r'=reset, 'q'=quit"  # 視窗名稱
DEFAULT_MODEL_PATH = "hand_landmarker.task"  # 手部地標模型檔案路徑
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_FPS = 30  # 估計攝影機 FPS，供 timestamp 使用

# 校準參數
CALIBRATION_MIN_DISTANCE_MM = 0.0  # 最小距離校準值（mm）
CALIBRATION_MAX_DISTANCE_MM = 130.0  # 最大距離校準值（mm）

# 目標關節點
TARGET_LANDMARK_1 = 4  # 拇指尖
TARGET_LANDMARK_2 = 8  # 食指尖

# MQTT 設定
MQTT_BROKER = "mqttgo.io"  # MQTT broker 位址
MQTT_PORT = 1883  # MQTT broker 連接埠
MQTT_TOPIC = "/tseng/mp_h_landmark_4to8_dist"  # MQTT publish topic
MQTT_QOS = 0  # MQTT QoS 等級(0: 至多一次, 1: 至少一次, 2: 僅一次)


# ==============================================================================
# 校準狀態列舉
# ==============================================================================


class CalibrationState(Enum):
    """校準狀態。"""
    WAIT_MIN = auto()  # 等待設定最小值（0mm）
    WAIT_MAX = auto()  # 等待設定最大值（130mm）
    CALIBRATED = auto()  # 已完成校準


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
    instruction_color_bgr: tuple[int, int, int] = (255, 255, 255)  # 指示文字顏色（白色）

    # 校準參數
    calibration_min_mm: float = CALIBRATION_MIN_DISTANCE_MM  # 最小距離校準值
    calibration_max_mm: float = CALIBRATION_MAX_DISTANCE_MM  # 最大距離校準值

    # MQTT 參數
    mqtt_broker: str = MQTT_BROKER  # MQTT broker 位址
    mqtt_port: int = MQTT_PORT  # MQTT broker 連接埠
    mqtt_topic: str = MQTT_TOPIC  # MQTT publish topic
    mqtt_qos: int = MQTT_QOS  # MQTT QoS 等級
    mqtt_enabled: bool = True  # 是否啟用 MQTT 發送


# ==============================================================================
# 手指距離偵測器類別（含兩點校準 + MQTT）
# ==============================================================================


class FingerDistanceDetectorCalibrated:
    """封裝 MediaPipe Tasks HandLandmarker 手指距離計算（含兩點校準 + MQTT），便於重複使用與後續擴充。"""

    def __init__(self, config: FingerDistanceConfig | None = None) -> None:
        self.config = config or FingerDistanceConfig()
        self._cap: cv2.VideoCapture | None = None
        self._landmarker = self._create_landmarker()
        
        # 校準狀態
        self._calibration_state = CalibrationState.WAIT_MIN
        self._min_distance_px: float | None = None  # 最小距離的像素值（對應 0mm）
        self._max_distance_px: float | None = None  # 最大距離的像素值（對應 130mm）
        
        # MQTT 發送計時器
        self._last_mqtt_send_time = 0.0  # 上次發送時間戳記
        self._mqtt_send_interval = 0.5  # 每 0.5 秒發送一次
        
        # MQTT 客戶端
        self._mqtt_client: mqtt.Client | None = None
        self._mqtt_connected = False
        if self.config.mqtt_enabled:
            self._setup_mqtt()

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

    def _setup_mqtt(self) -> None:
        """設定 MQTT 連線。"""
        try:
            # 建立 MQTT 客戶端
            self._mqtt_client = mqtt.Client(client_id="", protocol=mqtt.MQTTv311)
            
            # 設定連線回調函式
            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    self._mqtt_connected = True
                    print(f"MQTT 已連線到 {self.config.mqtt_broker}")
                else:
                    print(f"MQTT 連線失敗,錯誤碼: {rc}")
            
            def on_disconnect(client, userdata, rc):
                self._mqtt_connected = False
                print(f"MQTT 已斷線,錯誤碼: {rc}")
            
            self._mqtt_client.on_connect = on_connect
            self._mqtt_client.on_disconnect = on_disconnect
            
            # 連線到 MQTT broker
            print(f"正在連線到 MQTT broker: {self.config.mqtt_broker}:{self.config.mqtt_port}")
            self._mqtt_client.connect(self.config.mqtt_broker, self.config.mqtt_port, 60)
            self._mqtt_client.loop_start()  # 啟動背景執行緒處理網路事件
            
        except Exception as e:
            print(f"MQTT 設定失敗: {e}")
            self._mqtt_client = None

    def _publish_distance(self, distance_mm: float) -> None:
        """將距離值發送到 MQTT broker（每 2 秒發送一次）。"""
        if not self.config.mqtt_enabled or not self._mqtt_client or not self._mqtt_connected:
            return
        
        # 檢查是否已到達發送間隔
        current_time = time.time()
        if current_time - self._last_mqtt_send_time < self._mqtt_send_interval:
            return
        
        try:
            # 只發送距離數值
            payload = str(round(distance_mm, 2))
            
            # 發送訊息
            result = self._mqtt_client.publish(
                self.config.mqtt_topic,
                payload,
                qos=self.config.mqtt_qos
            )
            
            # 更新上次發送時間
            self._last_mqtt_send_time = current_time
            
            # 檢查發送結果(非阻塞)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"MQTT 發送失敗: {result.rc}")
            else:
                print(f"MQTT 已發送: {distance_mm:.2f} mm")
                
        except Exception as e:
            print(f"MQTT 發送錯誤: {e}")

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

    def _reset_calibration(self) -> None:
        """重置校準狀態。"""
        self._calibration_state = CalibrationState.WAIT_MIN
        self._min_distance_px = None
        self._max_distance_px = None
        print("校準已重置，請重新開始")

    def _set_min_calibration(self, distance_px: float) -> None:
        """設定最小距離校準點（0mm）。"""
        self._min_distance_px = distance_px
        self._calibration_state = CalibrationState.WAIT_MAX
        print(f"已設定最小距離: {distance_px:.2f} px = {self.config.calibration_min_mm} mm")
        print("請將手指張開至最大距離，然後按 'M'（Shift+m）鍵")

    def _set_max_calibration(self, distance_px: float) -> None:
        """設定最大距離校準點（130mm）。"""
        if self._min_distance_px is None:
            print("錯誤：請先設定最小距離（按 'm' 鍵）")
            return

        if distance_px <= self._min_distance_px:
            print(f"警告：最大距離 ({distance_px:.2f} px) 應大於最小距離 ({self._min_distance_px:.2f} px)")
            return

        self._max_distance_px = distance_px
        self._calibration_state = CalibrationState.CALIBRATED
        print(f"已設定最大距離: {distance_px:.2f} px = {self.config.calibration_max_mm} mm")
        print("校準完成！現在可以開始測量")

    def _calculate_calibrated_distance(self, distance_px: float) -> float:
        """根據校準點計算實際距離（mm）。"""
        if self._min_distance_px is None or self._max_distance_px is None:
            return 0.0

        # 線性插值：(distance_px - min_px) / (max_px - min_px) = (distance_mm - min_mm) / (max_mm - min_mm)
        px_range = self._max_distance_px - self._min_distance_px
        mm_range = self.config.calibration_max_mm - self.config.calibration_min_mm

        if px_range <= 0:
            return 0.0

        # 計算距離（mm）
        distance_mm = self.config.calibration_min_mm + (distance_px - self._min_distance_px) * (mm_range / px_range)
        
        # 限制範圍（避免超出校準範圍）
        distance_mm = max(self.config.calibration_min_mm, min(self.config.calibration_max_mm, distance_mm))
        
        return distance_mm

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

    def _draw_instruction_text(
        self,
        frame: np.ndarray,
    ) -> None:
        """在畫面上顯示校準指示文字。"""
        if self._calibration_state == CalibrationState.WAIT_MIN:
            text = "Step 1: Close fingers, press 'm' to set 0mm"
            color = (0, 0, 255)  # 紅色
        elif self._calibration_state == CalibrationState.WAIT_MAX:
            text = "Step 2: Open fingers, press 'M' to set 130mm"
            color = (0, 165, 255)  # 橘色
        else:
            text = "Calibrated! Press 'r' to reset"
            color = (0, 255, 0)  # 綠色

        # 顯示在畫面上方
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    def _draw_distance_text(
        self,
        frame: np.ndarray,
        distance_mm: float,
        distance_px: float,
    ) -> None:
        """在畫面上顯示距離數值。"""
        if self._calibration_state == CalibrationState.CALIBRATED:
            text = f"Distance: {distance_mm:.1f} mm"
        else:
            text = f"Current: {distance_px:.1f} px"

        # 顯示在畫面左側中間偏上
        cv2.putText(
            frame,
            text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale + 0.3,
            self.config.text_color_bgr,
            self.config.text_thickness,
        )

    def _process_and_draw(
        self,
        frame: np.ndarray,
        landmarks: list[Any],
    ) -> float:
        """處理左手地標，計算距離並繪製於畫面上，回傳當前像素距離。"""
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

        # 計算實際距離（如果已校準）
        distance_mm = 0.0
        if self._calibration_state == CalibrationState.CALIBRATED:
            distance_mm = self._calculate_calibrated_distance(distance_px)
            # 發送到 MQTT
            self._publish_distance(distance_mm)

        # 顯示距離數值
        self._draw_distance_text(frame, distance_mm, distance_px)

        return distance_px

    def run(self) -> None:
        """執行即時手指距離偵測（含兩點校準 + MQTT），顯示於視窗。"""
        self._cap = self._open_camera()
        frame_count = 0

        print(f"攝影機已開啟（ID: {self.config.camera_id}）")
        print("偵測左手，計算關節點 4（拇指尖）到關節點 8（食指尖）的距離")
        print("\n校準步驟：")
        print("1. 將手指併攏（讓關節點 4 和 8 最接近），按 'm' 鍵設定為 0mm")
        print("2. 將手指張開（讓關節點 4 和 8 最遠），按 'M'（Shift+m）鍵設定為 130mm")
        print("3. 校準完成後，即可正常測量並自動發送到 MQTT broker")
        print("\n按鍵說明：")
        print("- 'm' : 設定最小距離（0mm）")
        print("- 'M' : 設定最大距離（130mm）")
        print("- 'r' : 重新校準")
        print("- 'q' : 退出程式\n")

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
                current_distance_px = 0.0
                if left_hand_lm is not None:
                    current_distance_px = self._process_and_draw(frame, left_hand_lm)

                # 顯示校準指示
                self._draw_instruction_text(frame)

                # 顯示畫面
                cv2.imshow(WINDOW_NAME, frame)

                # 處理按鍵
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord("q"):
                    print("使用者中斷程式")
                    break
                elif key == ord("m"):
                    # 設定最小距離（0mm）
                    if left_hand_lm is not None and current_distance_px > 0:
                        self._set_min_calibration(current_distance_px)
                    else:
                        print("請確保偵測到左手")
                elif key == ord("M"):
                    # 設定最大距離（130mm）
                    if left_hand_lm is not None and current_distance_px > 0:
                        self._set_max_calibration(current_distance_px)
                    else:
                        print("請確保偵測到左手")
                elif key == ord("r"):
                    # 重新校準
                    self._reset_calibration()

        finally:
            # 關閉 MQTT 連線
            if self._mqtt_client:
                self._mqtt_client.loop_stop()
                self._mqtt_client.disconnect()
                print("MQTT 已斷線")
            
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
    detector = FingerDistanceDetectorCalibrated(config)
    detector.run()


if __name__ == "__main__":
    main()
