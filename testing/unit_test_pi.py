#!/usr/bin/env python3
import os
import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# =========================
# CONFIGURATION
# =========================
TEST_IMAGE = "test_capture.jpg"
TEST_VIDEO = "test_capture.h264"
VIDEO_DURATION_SEC = 3
MODEL_PATH = "models/fall_geom_lstm_builtin.tflite"

# =========================
# TRACK TEST RESULTS
# =========================
test_results = {}

# =========================
# CAMERA TEST CLASS
# =========================
class CameraTest:
    def __init__(self):
        self.camera = Picamera2()

    def test_preview(self):
        try:
            config = self.camera.create_preview_configuration()
            self.camera.configure(config)
            self.camera.start()

            frame = self.camera.capture_array()

            if frame is not None:
                print("✅ Preview test (headless) passed")
                test_results['Preview Test'] = "PASSED"
            else:
                print("❌ Preview test failed")
                test_results['Preview Test'] = "FAILED"

            self.camera.stop()

        except Exception as e:
            print(f"❌ Preview Test Failed: {e}")
            test_results['Preview Test'] = "FAILED"

    def test_image_capture(self):
        try:
            self.camera.start()
            self.camera.start_and_capture_file(TEST_IMAGE)
            self.camera.stop()

            if os.path.exists(TEST_IMAGE):
                print(f"✅ Image capture successful: {TEST_IMAGE}")
                test_results['Image Capture'] = "PASSED"
                os.remove(TEST_IMAGE)
            else:
                print(f"❌ Image capture failed: {TEST_IMAGE} not found")
                test_results['Image Capture'] = "FAILED"

        except Exception as e:
            print(f"❌ Image Capture Failed: {e}")
            test_results['Image Capture'] = "FAILED"

    def test_video_capture(self):
        try:
            self.camera.start()
            self.camera.start_and_record_video(TEST_VIDEO, duration=VIDEO_DURATION_SEC)
            self.camera.stop()

            if os.path.exists(TEST_VIDEO):
                print(f"✅ Video capture successful: {TEST_VIDEO}")
                test_results['Video Capture'] = "PASSED"
                os.remove(TEST_VIDEO)
            else:
                print(f"❌ Video capture failed: {TEST_VIDEO} not found")
                test_results['Video Capture'] = "FAILED"

        except Exception as e:
            print(f"❌ Video Capture Failed: {e}")
            test_results['Video Capture'] = "FAILED"

# =========================
# OPENCV TEST CLASS
# =========================
class OpenCVTest:
    @staticmethod
    def test_processing():
        try:
            dummy = 255 * np.ones((480, 640, 3), dtype=np.uint8)
            gray = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (224, 224))

            if resized.shape == (224, 224):
                print("✅ OpenCV processing successful")
                test_results['OpenCV Processing'] = "PASSED"
            else:
                print("❌ OpenCV processing failed: Unexpected shape")
                test_results['OpenCV Processing'] = "FAILED"

        except Exception as e:
            print(f"❌ OpenCV processing failed: {e}")
            test_results['OpenCV Processing'] = "FAILED"

# =========================
# MODEL LOADING TEST CLASS
# =========================
class ModelTest:
    @staticmethod
    def test_model_loading(model_path):
        try:
            interp = tflite.Interpreter(model_path=model_path)
            interp.allocate_tensors()

            print(f"✅ Model loaded successfully: {model_path}")
            test_results['Model Load'] = "PASSED"

        except Exception as e:
            print(f"❌ Model load failed: {e}")
            test_results['Model Load'] = "FAILED"

# =========================
# RUN ALL TESTS
# =========================
def run_tests():
    print("Starting Raspberry Pi 5 Unit Tests...\n")

    cam_test = CameraTest()
    cam_test.test_preview()
    cam_test.test_image_capture()
    cam_test.test_video_capture()

    OpenCVTest.test_processing()
    ModelTest.test_model_loading(MODEL_PATH)

    print("\n===== UNIT TEST SUMMARY =====")
    for test, result in test_results.items():
        print(f"{test}: {result}")

if __name__ == "__main__":
    run_tests()