
import cv2
import time
import numpy as np

from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor
from src.emotion_detector import detect_emotion


# Reduce OpenCV CPU thread usage
cv2.setNumThreads(1)


# Start webcam using DirectShow
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot access webcam!")
    exit()


# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Set MJPG format
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter.fourcc(*"MJPG")
)


# Check camera resolution
print("Camera Backend:", cap.getBackendName())
print(
    "Camera Resolution:",
    cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    "x",
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print("Live Emotion Detection Started!")
print("Press Q to stop.")


# Screen resolution
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


# Resize image without changing aspect ratio
def resize_with_aspect_ratio(image):

    h, w = image.shape[:2]

    scale = min(
        SCREEN_WIDTH / w,
        SCREEN_HEIGHT / h
    )

    new_width = int(w * scale)
    new_height = int(h * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Create black fullscreen background
    canvas = np.zeros(
        (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
        dtype=np.uint8
    )

    # Center image
    x = (SCREEN_WIDTH - new_width) // 2
    y = (SCREEN_HEIGHT - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


# Create fullscreen window
window_name = "Live Emotion Detection"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# Background emotion detection
executor = ThreadPoolExecutor(max_workers=1)

future = None
last_detection_time = 0

# Store previous emotions
emotion_history = deque(maxlen=1)

faces = []


while True:

    # Capture camera frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame!")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    current_time = time.monotonic()


    # Check completed emotion detection
    if future is not None and future.done():

        try:
            result = future.result()

            if result["success"] and result["faces"]:

                # Get first detected face
                face = result["faces"][0]

                # Save emotion history
                emotion_history.append({
                    "emotion": face["emotion"],
                    "confidence": face["confidence"],
                    "region": face["region"]
                })

                # Find most common emotion
                emotion_counts = Counter(
                    item["emotion"]
                    for item in emotion_history
                )

                stable_emotion = emotion_counts.most_common(1)[0][0]

                # Calculate average confidence
                matching = [
                    item for item in emotion_history
                    if item["emotion"] == stable_emotion
                ]

                avg_confidence = sum(
                    item["confidence"]
                    for item in matching
                ) / len(matching)

                # Latest face position
                latest_region = face["region"]

                faces = [{
                    "emotion": stable_emotion,
                    "confidence": round(avg_confidence, 2),
                    "region": latest_region
                }]

            else:
                emotion_history.clear()
                faces = []

        except Exception as e:
            print("Detection error:", e)

        future = None


    # Run emotion detection once per second
    if (
        future is None
        and current_time - last_detection_time >= 1.0
    ):

        future = executor.submit(
            detect_emotion,
            frame.copy()
        )

        last_detection_time = current_time


    # Draw face and emotion information
    for face in faces:

        emotion = face["emotion"]
        confidence = face["confidence"]

        x = face["region"]["x"]
        y = face["region"]["y"]
        w = face["region"]["w"]
        h = face["region"]["h"]


        # Draw face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        # Emotion label
        label = f"{emotion}: {confidence}%"

        font = cv2.FONT_HERSHEY_SIMPLEX

        (tw, th), baseline = cv2.getTextSize(
            label,
            font,
            0.7,
            2
        )

        label_y = y - 10

        if label_y - th - 10 < 0:
            label_y = y + th + 15


        # Label background
        cv2.rectangle(
            frame,
            (x, label_y - th - 10),
            (x + tw + 10, label_y + baseline),
            (0, 255, 0),
            -1
        )


        # Write emotion text
        cv2.putText(
            frame,
            label,
            (x + 5, label_y - 5),
            font,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


    # Preserve aspect ratio for fullscreen display
    display_frame = resize_with_aspect_ratio(frame)

    cv2.imshow(
        window_name,
        display_frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()

executor.shutdown(
    wait=False,
    cancel_futures=True
)

print("Camera stopped successfully!")
