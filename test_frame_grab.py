import cv2

# go2rtc re-serves the stream locally; connect via its own RTSP relay
url = "rtsp://localhost:8554/front_door"

cap = cv2.VideoCapture(url)
ret, frame = cap.read()

if ret:
    print(f"Success! Frame shape: {frame.shape}")
    cv2.imwrite("test_frame.jpg", frame)
    print("Saved test_frame.jpg")
else:
    print("Failed to grab frame")

cap.release()