import sys
import platform

def print_section(title):
    """列印區段標題"""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def check_python_version():
    """檢查 Python 版本"""
    print_section("Python 版本資訊")
    print(f"Python 版本: {sys.version}")
    print(f"Python 版本號: {platform.python_version()}")
    print(f"Python 編譯器: {platform.python_compiler()}")
    print(f"平台: {platform.platform()}")

def check_mediapipe():
    """檢查 MediaPipe 版本"""
    print_section("MediaPipe 版本資訊")
    try:
        import mediapipe as mp
        print(f"MediaPipe 版本: {mp.__version__}")
        print(f"MediaPipe 安裝路徑: {mp.__file__}")
    except ImportError as e:
        print(f"❌ MediaPipe 未安裝: {e}")
        return False
    return True

def check_opencv():
    """檢查 OpenCV 版本與建構資訊"""
    print_section("OpenCV 版本資訊")
    try:
        import cv2
        print(f"OpenCV 版本: {cv2.__version__}")
        print(f"OpenCV 安裝路徑: {cv2.__file__}")
        
        # 檢查建構資訊
        build_info = cv2.getBuildInformation()
        
        # 提取 CUDA 相關資訊
        print(f"\n--- 建構資訊 ---")
        for line in build_info.split('\n'):
            if 'CUDA' in line or 'cuDNN' in line or 'GPU' in line:
                print(line.strip())
        
        return True
    except ImportError as e:
        print(f"❌ OpenCV 未安裝: {e}")
        return False

def check_cuda():
    """檢查 CUDA 支援與 GPU 資訊"""
    print_section("CUDA & GPU 資訊")
    
    # 檢查 OpenCV 的 CUDA 支援
    try:
        import cv2
        cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if cuda_enabled:
            print(f"✅ OpenCV CUDA 已啟用")
            print(f"CUDA 裝置數量: {cv2.cuda.getCudaEnabledDeviceCount()}")
            
            for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
                print(f"\n裝置 {i} 資訊:")
                try:
                    print(f"  裝置名稱: {cv2.cuda.getDevice()}")
                except:
                    print(f"  無法取得裝置名稱")
        else:
            print("❌ OpenCV 未啟用 CUDA 支援")
    except AttributeError:
        print("❌ OpenCV 未編譯 CUDA 模組")
    except Exception as e:
        print(f"❌ 檢查 OpenCV CUDA 時發生錯誤: {e}")
    
    # 檢查 PyTorch CUDA
    print("\n--- PyTorch CUDA 支援 ---")
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"GPU 裝置數量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  名稱: {torch.cuda.get_device_name(i)}")
                print(f"  記憶體總量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"  CUDA 核心數: {torch.cuda.get_device_properties(i).multi_processor_count}")
                print(f"  計算能力: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    except ImportError:
        print("⚠️  PyTorch 未安裝")
    except Exception as e:
        print(f"❌ 檢查 PyTorch CUDA 時發生錯誤: {e}")
    
    # 檢查 TensorFlow GPU
    print("\n--- TensorFlow GPU 支援 ---")
    try:
        import tensorflow as tf
        print(f"TensorFlow 版本: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"GPU 裝置數量: {len(gpus)}")
        if gpus:
            print("✅ TensorFlow GPU 支援已啟用")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    print(f"    詳細資訊: {gpu_details}")
                except:
                    pass
        else:
            print("❌ 未偵測到 GPU 裝置")
    except ImportError:
        print("⚠️  TensorFlow 未安裝")
    except Exception as e:
        print(f"❌ 檢查 TensorFlow GPU 時發生錯誤: {e}")

def check_system_info():
    """顯示系統資訊"""
    print_section("系統資訊")
    print(f"作業系統: {platform.system()} {platform.release()}")
    print(f"架構: {platform.machine()}")
    print(f"處理器: {platform.processor()}")
    
    # 檢查記憶體（如果 psutil 可用）
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"總記憶體: {mem.total / 1024**3:.2f} GB")
        print(f"可用記憶體: {mem.available / 1024**3:.2f} GB")
        print(f"CPU 核心數: {psutil.cpu_count(logical=False)} (物理)")
        print(f"CPU 執行緒數: {psutil.cpu_count(logical=True)} (邏輯)")
    except ImportError:
        print("⚠️  psutil 未安裝，無法顯示記憶體資訊")

def main():
    """主函式"""
    print("\n" + "="*50)
    print(" 環境檢測程式")
    print("="*50)
    
    # 執行所有檢查
    check_python_version()
    check_mediapipe()
    check_opencv()
    check_cuda()
    check_system_info()
    
    print("\n" + "="*50)
    print(" 檢測完成")
    print("="*50)

if __name__ == "__main__":
    main()
