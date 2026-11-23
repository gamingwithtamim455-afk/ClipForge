# Bundled MediaPipe models

Downloaded from Google's MediaPipe model storage
(https://storage.googleapis.com/mediapipe-models/...), Apache License 2.0:

| File | Model | Used for |
|---|---|---|
| `blaze_face_short_range.tflite` | BlazeFace short-range (float16) | face detection |
| `face_landmarker.task` | Face Landmarker (float16) | 478 facial landmarks incl. lips |

MediaPipe 1.2.3 ships only the Tasks API (`mediapipe.tasks.python.vision`);
the legacy `mediapipe.solutions` module is gone, so the face tracker uses
`FaceDetector` + `FaceLandmarker` in `IMAGE` mode on sampled frames.

Linux dev note: the MediaPipe wheel links against `libGLESv2.so.2`
(`apt install libgles2`). The Windows wheel does not need it.
