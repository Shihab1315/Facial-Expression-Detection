
import cv2
import numpy as np
from src.emotion_detector import detect_emotion

# Load image
image = cv2.imread("test.jpg")

if image is None:
    print("Image not found!")
    exit()

# Detect emotions
result = detect_emotion(image)

if not result["success"] or not result["faces"]:
    print(result.get("error", "No face detected!"))
    exit()

# Emotion display order
emotion_order = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]

original_height, original_width = image.shape[:2]

# Resize image while preserving aspect ratio
max_width = 900
max_height = 700

scale = min(
    max_width / original_width,
    max_height / original_height,
    1.0
)

new_width = int(original_width * scale)
new_height = int(original_height * scale)

image = cv2.resize(
    image,
    (new_width, new_height),
    interpolation=cv2.INTER_AREA
)

# Process every detected face
for index, face in enumerate(result["faces"]):

    emotion = face["emotion"]
    confidence = face["confidence"]
    emotions = face["emotions"]

    # Print all emotion scores
    print(f"\nFace {index + 1}")
    print("-" * 30)
    print("Dominant Emotion:", emotion)
    print("Confidence:", confidence, "%")

    print("\nAll Emotion Scores:")

    for name in emotion_order:
        score = emotions.get(name, 0)
        print(f"{name.capitalize():10}: {score:.2f}%")

    # Get face coordinates
    region = face["region"]

    x = int(region["x"] * scale)
    y = int(region["y"] * scale)
    w = int(region["w"] * scale)
    h = int(region["h"] * scale)

    # Draw face rectangle
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Draw dominant emotion label
    label = f"{emotion.upper()} ({confidence:.1f}%)"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )

    # Keep label inside image boundaries
    label_y = y - 10

    if label_y - text_height - 10 < 0:
        label_y = y + text_height + 15

    label_y = min(label_y, new_height - baseline - 2)

    # Background for label
    cv2.rectangle(
        image,
        (x, label_y - text_height - 10),
        (min(x + text_width + 10, new_width), label_y + baseline),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        image,
        label,
        (x + 5, label_y - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )


# Create emotion score panel
panel_width = 340
panel_height = max(new_height, len(result["faces"]) * 235 + 30)

panel = np.zeros(
    (panel_height, panel_width, 3),
    dtype=np.uint8
)

# Panel heading
cv2.putText(
    panel,
    "EMOTION ANALYSIS",
    (15, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

# Display scores for every face
for index, face in enumerate(result["faces"]):

    emotions = face["emotions"]
    emotion = face["emotion"]

    start_y = 75 + index * 235

    cv2.putText(
        panel,
        f"Face {index + 1}: {emotion.upper()}",
        (15, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    for i, name in enumerate(emotion_order):

        score = emotions.get(name, 0)
        y_pos = start_y + 30 + i * 25

        text = f"{name.capitalize():10} {score:5.2f}%"

        cv2.putText(
            panel,
            text,
            (20, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

# Combine image and score panel
final_image = np.zeros(
    (max(new_height, panel_height), new_width + panel_width, 3),
    dtype=np.uint8
)

final_image[:new_height, :new_width] = image
final_image[:panel_height, new_width:] = panel

# Display result
cv2.imshow(
    "Facial Expression Detection",
    final_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
