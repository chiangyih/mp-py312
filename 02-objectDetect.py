"""使用 MediaPipe Tasks 進行單張圖片的簡易物件偵測，結果輸出於終端機。"""

from pathlib import Path
from urllib.request import urlretrieve

import mediapipe as mp


# 預設圖片與模型路徑
IMAGE_PATH = "image.jpg"
MODEL_PATH = "efficientdet_lite0.tflite" # 模型檔案路徑
MODEL_URL = (
	"https://storage.googleapis.com/mediapipe-models/object_detector/"
	"efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)


def ensure_file_exists(path: str, download_url: str | None = None):
	"""確保模型檔案存在，必要時從指定網址下載。"""
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


def validate_image(path: str) -> Path:
	"""檢查圖片是否存在。"""
	image_path = Path(path)
	if not image_path.exists():
		raise FileNotFoundError(f"找不到圖片檔案: {image_path}")
	return image_path


def describe_detection(idx: int, detection, image_width: int, image_height: int):
	"""將偵測結果轉為易讀文字並印出。"""
	box = detection.bounding_box
	x_min = max(0, box.origin_x)
	y_min = max(0, box.origin_y)
	box_width = box.width
	box_height = box.height

	# 顯示最可能的分類與分數
	if detection.categories:
		top_category = detection.categories[0]
		label_text = top_category.category_name or "(no label)"
		score = top_category.score
	else:
		label_text = "(no label)"
		score = 0.0

	print(
		f"偵測 {idx}: label={label_text}, score={score:.2f}, "
		f"box=(x={x_min}, y={y_min}, w={box_width}, h={box_height})"
	)


def run_detection(image_path: str, model_path: str):
	"""執行物件偵測並在終端機顯示結果。"""
	image_file = validate_image(image_path)
	ensure_file_exists(model_path, MODEL_URL)

	BaseOptions = mp.tasks.BaseOptions
	vision = mp.tasks.vision

	options = vision.ObjectDetectorOptions(
		base_options=BaseOptions(model_asset_path=str(model_path)),
		score_threshold=0.5,
		max_results=5,
		running_mode=vision.RunningMode.IMAGE,
	)

	detector = vision.ObjectDetector.create_from_options(options)

	# 直接從檔案建立 mp.Image，省去 OpenCV 轉換
	mp_image = mp.Image.create_from_file(str(image_file))
	result = detector.detect(mp_image)

	detections = result.detections or []
	print(f"檔案: {image_file}")
	print(f"偵測數量: {len(detections)}")

	if not detections:
		return

	for idx, detection in enumerate(detections, start=1):
		describe_detection(idx, detection, mp_image.width, mp_image.height)


def main():
	run_detection(IMAGE_PATH, MODEL_PATH)


if __name__ == "__main__":
	main()
