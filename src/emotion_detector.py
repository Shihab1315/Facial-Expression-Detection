
from deepface import DeepFace
from typing import Any, cast


def detect_emotion(image):

    try:
        results = DeepFace.analyze(
            img_path=image,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True,
            silent=True
        )

        detected_faces = []

        for face in results:

            emotion_scores = {
                emotion: round(float(score), 2)
                for emotion, score in cast(dict[str, Any], face)["emotion"].items()
            }

            emotion = cast(dict[str, Any], face)["dominant_emotion"]
            confidence = emotion_scores[emotion]

            detected_faces.append({
                "emotion": emotion,
                "confidence": confidence,
                "emotions": emotion_scores,
                "region": cast(dict[str, Any], face)["region"]
            })

        return {
            "success": True,
            "faces": detected_faces
        }

    except ValueError:
        return {
            "success": False,
            "faces": [],
            "error": "No face detected."
        }

    except Exception as e:
        return {
            "success": False,
            "faces": [],
            "error": str(e)
        }
