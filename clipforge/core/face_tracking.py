"""Face tracking (professional): detection, identity tracking, active-speaker
selection and the smoothed crop path.

Rules (build spec §5):
- Detect faces + landmarks (MediaPipe), track identities across frames.
- Active speaker = lip movement + audio activity; hold a speaker >= 1.5 s
  before switching; reset the hold at scene cuts.
- Optimization-based smoothed crop path (penalized least squares: follow
  the face, penalize acceleration — no jitter), headroom, never cut off
  the head.
- Crop coordinates come from native-resolution frames, before scaling.
- Auto-fallback to a centred FILL crop when no face is found: the
  rendered clip always fills the whole output frame (an earlier
  "blur-fit" fallback fitted the whole 16:9 picture inside the 9:16
  canvas over a blurred background — letterboxed output the user
  rejected; that mode was removed in v0.2.1).
- Two people sharing a clip (both present through the range) get a
  two-up split-screen path when the Experimental setting is on.
- Auto-QA: face inside the crop in >= 99% of sampled frames; otherwise
  retry with a stricter path and mark the clip for the Edit step.
"""

from __future__ import annotations

import math
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .ffmpeg import extract_audio, probe

__all__ = [
    "CropPath",
    "FaceBox",
    "detect_scene_cuts",
    "dominant_identity_pair",
    "face_inside_crop_fraction",
    "models_dir",
    "scene_cut_flags",
    "smooth_path_1d",
    "track_faces",
]

#: Minimum hold time on one speaker before a switch (seconds).
MIN_SPEAKER_HOLD_S = 1.5
#: Default smoothing strength (higher = smoother, more lag).
DEFAULT_LAMBDA = 8.0
#: Default max pan speed of the crop window (native px / second).
DEFAULT_MAX_SPEED_PX_S = 400.0
#: Fraction of the crop height kept clear above the head.
HEADROOM_FRACTION = 0.15
#: QA pass threshold: face inside the crop in >= 99% of sampled frames.
QA_THRESHOLD = 0.99
#: PySceneDetect content-change threshold (0-100; 27 is the library
#: default and a good general value for hard cuts).
SCENE_CUT_THRESHOLD = 27.0
#: Detection runs on frames downscaled to at most this width. Detection
#: output is normalised coordinates, so the crop math (native pixels) is
#: unaffected — but a 4K frame held for detection costs ~16x the memory
#: and CPU of a 960 px one, and a clip samples ~200 frames.
DETECT_MAX_WIDTH = 960
#: Split screen (Settings > Experimental): the second-most-present
#: person must appear in at least this share of the samples that contain
#: any face for the clip to count as two-person — below that they are a
#: passer-by and the clip keeps the single tracked crop.
SPLIT_MIN_SECOND_FRACTION = 0.30
#: Split screen is only considered with at least this many face samples
#: (very short or mostly faceless clips keep the single crop).
SPLIT_MIN_FACE_SAMPLES = 5


@dataclass
class FaceBox:
    """A detected face in one frame (normalized 0..1 coordinates)."""

    x: float  # left
    y: float  # top
    width: float
    height: float
    identity: int = 0
    speaking_score: float = 0.0
    lip_open: float = 0.0  # mouth openness, fraction of face height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2

    @property
    def cy(self) -> float:
        return self.y + self.height / 2

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height


@dataclass
class CropPath:
    """Smoothed crop rectangle over time for one clip.

    ``boxes`` are the crop rectangles in native pixels (x, y, width,
    height), one per entry in ``times`` (seconds, clip-relative).
    ``faces`` holds the active face per sample (``None`` when no face was
    detected at that time).

    In ``"split"`` mode (two-person split screen) ``boxes``/``faces``
    describe the TOP pane and ``boxes_b``/``faces_b`` the BOTTOM pane;
    each pane's crop has the output aspect at half height.
    """

    times: list[float] = field(default_factory=list)
    boxes: list[FaceBox] = field(default_factory=list)
    faces: list[FaceBox | None] = field(default_factory=list)
    mode: str = "track"  # "track" | "split"
    qa_face_inside_fraction: float = 1.0
    qa_passed: bool = True
    needs_review: bool = False  # True when QA failed even after retry
    fallback_fraction: float = 0.0  # share of samples with no face
    boxes_b: list[FaceBox] = field(default_factory=list)  # split: bottom pane
    faces_b: list[FaceBox | None] = field(default_factory=list)


# ----------------------------------------------------------------------
# Pure helpers (unit-testable without MediaPipe)
# ----------------------------------------------------------------------

def smooth_path_1d(
    targets: list[float],
    times: list[float],
    *,
    lam: float = DEFAULT_LAMBDA,
    max_speed: float = DEFAULT_MAX_SPEED_PX_S,
) -> list[float]:
    """Smooth a 1-D path with penalized least squares.

    Minimizes ``sum((x - target)^2) + lam * sum((d2x)^2)`` where ``d2x``
    is the second difference (acceleration). The system matrix is
    ``I + lam * D^T D`` with D the second-difference operator; solved
    directly (sample counts are small, a few hundred at most). A velocity
    clamp pass then limits the pan speed.
    """
    n = len(targets)
    if n == 0:
        return []
    if n < 3:
        return list(targets)
    target = np.asarray(targets, dtype=float)
    # Second-difference operator D (n-2 x n).
    d = np.zeros((n - 2, n))
    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    a = np.eye(n) + lam * (d.T @ d)
    try:
        smooth = np.linalg.solve(a, target)
    except np.linalg.LinAlgError:
        smooth = target.copy()
    # Velocity clamp: limit |dx/dt| to max_speed, forward + backward pass.
    times_arr = np.asarray(times, dtype=float)
    for _ in range(2):
        for i in range(1, n):
            dt = max(times_arr[i] - times_arr[i - 1], 1e-6)
            limit = max_speed * dt
            delta = smooth[i] - smooth[i - 1]
            if delta > limit:
                smooth[i] = smooth[i - 1] + limit
            elif delta < -limit:
                smooth[i] = smooth[i - 1] - limit
        for i in range(n - 2, -1, -1):
            dt = max(times_arr[i + 1] - times_arr[i], 1e-6)
            limit = max_speed * dt
            delta = smooth[i + 1] - smooth[i]
            if delta > limit:
                smooth[i] = smooth[i + 1] - limit
            elif delta < -limit:
                smooth[i] = smooth[i + 1] + limit
    return [float(v) for v in smooth]


def face_inside_crop_fraction(
    faces: list[FaceBox | None],
    crops: list[FaceBox],
    *,
    margin: float = 0.02,
) -> float:
    """Fraction of samples where the detected face is inside the crop.

    ``faces``/``crops`` are normalized 0..1 boxes; samples with no face
    are skipped. ``margin`` shrinks the crop slightly (inset) so the QA
    is strict about the face touching the edge.
    """
    checked = 0
    inside = 0
    for face, crop in zip(faces, crops):
        if face is None:
            continue
        checked += 1
        mx = margin * crop.width
        my = margin * crop.height
        if (
            face.x >= crop.x + mx
            and face.right <= crop.right - mx
            and face.y >= crop.y + my
            and face.bottom <= crop.bottom - my
        ):
            inside += 1
    return inside / checked if checked else 1.0


def _iou(a: FaceBox, b: FaceBox) -> float:
    """Intersection over union of two normalized boxes."""
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.right, b.right)
    y1 = min(a.bottom, b.bottom)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------
# MediaPipe pipeline
# ----------------------------------------------------------------------

def _audio_rms_windows(wav_path: str, times: list[float], window_s: float = 0.3) -> list[float]:
    """RMS audio level around each sample time (0.0 when unreadable)."""
    try:
        with wave.open(wav_path, "rb") as wav:
            rate = wav.getframerate()
            frames = wav.getnframes()
            raw = wav.readframes(frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    except (wave.Error, OSError, EOFError):
        return [0.0] * len(times)
    if len(samples) == 0:
        return [0.0] * len(times)
    out: list[float] = []
    half = int(window_s * rate / 2)
    for t in times:
        center = int(t * rate)
        lo = max(0, center - half)
        hi = min(len(samples), center + half)
        window = samples[lo:hi]
        out.append(float(np.sqrt(np.mean(window ** 2))) if len(window) else 0.0)
    return out


def models_dir() -> Path:
    """Directory holding the bundled MediaPipe model files."""
    return Path(__file__).resolve().parent.parent / "assets" / "models"


def _detect_faces(
    detector, landmarker, rgb: np.ndarray, frame_w: int, frame_h: int
) -> tuple[list[FaceBox], list[tuple[float, float, float]]]:
    """Detect faces + lip openness in one RGB frame.

    Returns ``(boxes, meshes)`` where ``meshes`` is a list of
    ``(center_x, center_y, lip_open)`` (normalized) for each landmarked
    face, used to attach lip movement to the nearest detected box.
    """
    import mediapipe as mp

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    boxes: list[FaceBox] = []
    for det in detector.detect(image).detections:
        bbox = det.bounding_box
        x = max(0.0, bbox.origin_x / frame_w)
        y = max(0.0, bbox.origin_y / frame_h)
        w = min(1.0 - x, bbox.width / frame_w)
        h = min(1.0 - y, bbox.height / frame_h)
        if w > 0 and h > 0:
            boxes.append(FaceBox(x=x, y=y, width=w, height=h))
    meshes: list[tuple[float, float, float]] = []
    for landmarks in landmarker.detect(image).face_landmarks:
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        lip = abs(landmarks[14].y - landmarks[13].y)  # lower - upper lip
        meshes.append((sum(xs) / len(xs), sum(ys) / len(ys), lip))
    return boxes, meshes


def detect_scene_cuts(
    video_path: str,
    *,
    start: float = 0.0,
    end: float | None = None,
    threshold: float = SCENE_CUT_THRESHOLD,
) -> list[float]:
    """Detect scene cuts in a video with PySceneDetect (ContentDetector).

    Returns cut times in **video-absolute seconds**, filtered to
    ``[start, end]``. Returns ``[]`` when PySceneDetect is unavailable
    or the video cannot be read — the caller then falls back to the
    frame-diff detector, so a missing/broken PySceneDetect can never
    break face tracking.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except Exception:  # noqa: BLE001 - optional dependency; fallback below
        return []
    try:
        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=threshold))
        if start > 0:
            # Seek first: without this the detector scans the whole file
            # from 0 even when only a later range is needed — on a 1-hour
            # video that took ~10 min to scan for a 40s range.
            try:
                video.seek(start)
            except Exception:  # noqa: BLE001 - seek unsupported; full scan
                pass
        manager.detect_scenes(video, end_time=end)
        # Cuts are the start of every scene after the first.
        scenes = manager.get_scene_list()
        cuts = [start.seconds for start, _end in scenes[1:]]
    except Exception:  # noqa: BLE001 - unreadable video; fallback below
        return []
    if end is None:
        return [c for c in cuts if c >= start]
    return [c for c in cuts if start <= c <= end]


def scene_cut_flags(
    sample_times: list[float],
    cut_times: list[float],
    *,
    start: float = 0.0,
) -> list[bool]:
    """Map cut times onto sampled frames.

    ``sample_times`` are clip-relative; ``cut_times`` are video-absolute
    (as returned by :func:`detect_scene_cuts`). Sample ``i`` is flagged
    when a cut falls after the previous sample and at or before it —
    i.e. that frame is the first frame of a new scene, so the speaker
    hold resets there.
    """
    flags: list[bool] = []
    cuts = sorted(c - start for c in cut_times)
    for i, t in enumerate(sample_times):
        prev = sample_times[i - 1] if i > 0 else 0.0
        flags.append(any(prev < c <= t for c in cuts))
    return flags


def _frame_diff_scene_cuts(
    frames: list[np.ndarray | None],
) -> list[bool]:
    """Fallback scene-cut detector: mean absolute difference of
    downscaled grayscale frames (threshold 0.15). Used only when
    PySceneDetect is unavailable or found no cuts."""
    import cv2

    scene_cut: list[bool] = []
    prev_gray: np.ndarray | None = None
    for frame in frames:
        if frame is None:
            scene_cut.append(False)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 90))
        if prev_gray is not None:
            diff = float(np.mean(np.abs(
                small.astype(np.float32) - prev_gray.astype(np.float32)
            ))) / 255.0
            scene_cut.append(diff > 0.15)
        else:
            scene_cut.append(False)
        prev_gray = small
    return scene_cut


def _detect_faces_in_frames(
    frames: list[np.ndarray | None],
    frame_w: int,
    frame_h: int,
    *,
    min_detection_confidence: float = 0.5,
    scene_cuts: list[bool] | None = None,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> tuple[list[list[FaceBox]], list[bool]]:
    """Run detection + landmarks over the sampled frames.

    Returns per-sample face boxes (with identity + lip openness filled
    in) and per-sample scene-cut flags. ``scene_cuts`` (when given, one
    flag per frame, e.g. from PySceneDetect via :func:`scene_cut_flags`)
    is used as-is; otherwise the frame-diff fallback runs.

    ``progress`` receives the fraction of frames processed (0..1) and
    ``cancel`` is polled per frame — a long scan must stay interruptible
    (v0.2.1: cancel used to wait for the whole scan to finish).
    """
    import cv2
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    detector = vision.FaceDetector.create_from_options(
        vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=str(
                    models_dir() / "blaze_face_short_range.tflite"
                )
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
    )
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(models_dir() / "face_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=4,
        )
    )
    try:
        detected: list[list[FaceBox]] = []
        if scene_cuts is not None and len(scene_cuts) == len(frames):
            scene_cut = list(scene_cuts)
        else:
            scene_cut = _frame_diff_scene_cuts(frames)
        next_identity = 1
        prev_faces: list[FaceBox] = []
        total_frames = max(len(frames), 1)
        for frame_i, (frame, _is_cut) in enumerate(zip(frames, scene_cut)):
            if cancel is not None and cancel():
                raise RuntimeError("face tracking cancelled")
            if progress is not None:
                progress((frame_i + 1) / total_frames)
            if frame is None:
                detected.append([])
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, meshes = _detect_faces(detector, landmarker, rgb, frame_w, frame_h)
            for box in boxes:
                # Identity: IoU match against the previous sample's faces.
                best, best_iou = -1, 0.25
                for k, prev in enumerate(prev_faces):
                    iou = _iou(box, prev)
                    if iou > best_iou:
                        best, best_iou = k, iou
                if best >= 0:
                    box.identity = prev_faces[best].identity
                else:
                    box.identity = next_identity
                    next_identity += 1
                # Lip openness: nearest landmarked face by center distance.
                best_d, best_lip = float("inf"), 0.0
                for mx, my, lip in meshes:
                    dist = math.hypot(mx - box.cx, my - box.cy)
                    if dist < best_d:
                        best_d, best_lip = dist, lip
                box.lip_open = best_lip / max(box.height, 1e-6)
            detected.append(boxes)
            prev_faces = boxes
        return detected, scene_cut
    finally:
        # Explicit close: interpreter-teardown order can otherwise raise
        # inside the MediaPipe dispatcher (mediapipe 1.2.3).
        detector.close()
        landmarker.close()


def _project_to_contain(
    crops_x: list[float],
    crops_y: list[float],
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
    faces: list[FaceBox | None],
) -> tuple[list[float], list[float]]:
    """Shift each crop rect the minimum amount so the face is inside it.

    Used as the strict-retry pass: guarantees QA improvement unless the
    face is larger than the crop (then it is flagged for review).
    """
    xs = list(crops_x)
    ys = list(crops_y)
    for i, face in enumerate(faces):
        if face is None:
            continue
        fx0, fy0 = face.x * frame_w, face.y * frame_h
        fx1, fy1 = face.right * frame_w, face.bottom * frame_h
        # Horizontal: keep the whole face inside [x, x + crop_w].
        if fx0 < xs[i]:
            xs[i] = fx0
        if fx1 > xs[i] + crop_w:
            xs[i] = fx1 - crop_w
        if fy0 < ys[i]:
            ys[i] = fy0
        if fy1 > ys[i] + crop_h:
            ys[i] = fy1 - crop_h
        xs[i] = min(max(xs[i], 0.0), float(frame_w - crop_w))
        ys[i] = min(max(ys[i], 0.0), float(frame_h - crop_h))
    return xs, ys


def dominant_identity_pair(
    detected: list[list[FaceBox]],
    *,
    min_second_fraction: float = SPLIT_MIN_SECOND_FRACTION,
    min_face_samples: int = SPLIT_MIN_FACE_SAMPLES,
) -> tuple[int, int] | None:
    """The two dominant face identities in a clip, or ``None``.

    ``detected`` is the per-sample face list from detection. Counts the
    samples each identity appears in; returns ``(first, second)`` (most
    present first) when the second person still covers at least
    ``min_second_fraction`` of the samples that contain any face — two
    people genuinely sharing the clip (a two-shot or a conversation),
    not a passer-by crossing one shot.
    """
    counts: dict[int, int] = {}
    face_samples = 0
    for boxes in detected:
        if not boxes:
            continue
        face_samples += 1
        for identity in {box.identity for box in boxes}:
            counts[identity] = counts.get(identity, 0) + 1
    if face_samples < min_face_samples or len(counts) < 2:
        return None
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    first, second = ranked[0][0], ranked[1][0]
    if counts[second] / face_samples < min_second_fraction:
        return None
    return first, second


def _identity_series(
    detected: list[list[FaceBox]], identity: int
) -> list[FaceBox | None]:
    """One identity's face per sample (``None`` where it is absent)."""
    return [
        next((box for box in boxes if box.identity == identity), None)
        for boxes in detected
    ]


def _crop_targets(
    faces: list[FaceBox | None],
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
) -> tuple[list[float], list[float]]:
    """Per-sample crop targets (native px) following ``faces``.

    The crop is centred on the face with the headroom rule (clear space
    above the head, never cutting off the head or the chin when they
    fit). Samples with no face hold the previous target (centre when
    there is none yet).
    """
    target_x: list[float] = []
    target_y: list[float] = []
    for face in faces:
        if face is None:
            target_x.append(target_x[-1] if target_x else (frame_w - crop_w) / 2)
            target_y.append(target_y[-1] if target_y else (frame_h - crop_h) / 2)
            continue
        fx0, fy0 = face.x * frame_w, face.y * frame_h
        fx1, fy1 = face.right * frame_w, face.bottom * frame_h
        x = face.cx * frame_w - crop_w / 2
        x = min(max(x, 0.0), float(frame_w - crop_w))
        y = fy0 - HEADROOM_FRACTION * crop_h
        y = min(y, fy0)  # never cut off the top of the head
        y = max(y, fy1 - crop_h)  # never cut off the chin when it fits
        y = min(max(y, 0.0), float(frame_h - crop_h))
        target_x.append(x)
        target_y.append(y)
    return target_x, target_y


def _person_crop_boxes(
    faces: list[FaceBox | None],
    times: list[float],
    crop_w: int,
    crop_h: int,
    frame_w: int,
    frame_h: int,
) -> tuple[list[FaceBox], float, bool]:
    """Smoothed crop boxes + QA for one person's face series.

    Mirrors the single-speaker flow: smooth the targets, check the face
    is inside the crop, and on failure retry stricter with containment
    projection. Returns ``(boxes, qa_fraction, qa_passed)``.
    """
    def build(lam: float, max_speed: float, project: bool):
        target_x, target_y = _crop_targets(faces, crop_w, crop_h, frame_w, frame_h)
        xs = smooth_path_1d(target_x, times, lam=lam, max_speed=max_speed)
        ys = smooth_path_1d(target_y, times, lam=lam, max_speed=max_speed)
        if project:
            xs, ys = _project_to_contain(
                xs, ys, crop_w, crop_h, frame_w, frame_h, faces
            )
        boxes = [
            FaceBox(x=x, y=y, width=float(crop_w), height=float(crop_h))
            for x, y in zip(xs, ys)
        ]
        norm_crops = [
            FaceBox(x=b.x / frame_w, y=b.y / frame_h,
                    width=b.width / frame_w, height=b.height / frame_h)
            for b in boxes
        ]
        return boxes, face_inside_crop_fraction(faces, norm_crops)

    boxes, frac = build(DEFAULT_LAMBDA, DEFAULT_MAX_SPEED_PX_S, False)
    if frac >= QA_THRESHOLD:
        return boxes, frac, True
    boxes, frac = build(DEFAULT_LAMBDA / 4, DEFAULT_MAX_SPEED_PX_S / 2.5, True)
    return boxes, frac, frac >= QA_THRESHOLD


def _split_crop_path(
    pair: tuple[int, int],
    detected: list[list[FaceBox]],
    times: list[float],
    aspect: tuple[int, int],
    frame_w: int,
    frame_h: int,
) -> CropPath | None:
    """Two-up split-screen crop path for a two-person clip, or ``None``.

    Each pane is the output aspect at half height, so one person's crop
    has aspect ``2*aw : ah``. Returns ``None`` when either pane fails QA
    even after the strict retry — the caller then keeps the normal
    single-speaker tracked crop.
    """
    aw, ah = aspect
    crop_w = int(frame_h * 2 * aw / ah)
    crop_h = frame_h
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = int(frame_w * ah / (2 * aw))
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    series = [_identity_series(detected, identity) for identity in pair]
    built = [
        _person_crop_boxes(s, times, crop_w, crop_h, frame_w, frame_h)
        for s in series
    ]
    if not all(passed for _boxes, _frac, passed in built):
        return None
    (boxes_a, frac_a, _), (boxes_b, frac_b, _) = built
    no_face = (
        sum(1 for boxes in detected if not boxes) / max(len(detected), 1)
    )
    return CropPath(
        times=list(times),
        boxes=boxes_a,
        faces=series[0],
        mode="split",
        boxes_b=boxes_b,
        faces_b=series[1],
        qa_face_inside_fraction=min(frac_a, frac_b),
        qa_passed=True,
        needs_review=False,
        fallback_fraction=no_face,
    )


def track_faces(
    video_path: str,
    *,
    start: float,
    end: float,
    aspect: tuple[int, int] = (9, 16),
    sample_fps: float = 5.0,
    min_detection_confidence: float = 0.5,
    split_screen: bool = True,
    progress: Callable[[float], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> CropPath:
    """Compute the smoothed crop path for a clip range.

    Samples frames at ``sample_fps`` between ``start`` and ``end``,
    detects faces + lip landmarks, picks the active speaker, and returns
    a smoothed native-resolution crop path with a QA verdict. When
    ``split_screen`` is on (Settings > Experimental) and two people
    genuinely share the clip, the path is a two-up split instead — see
    :func:`dominant_identity_pair`.

    ``progress`` receives the scan fraction (0..1: frame sampling is the
    first half, detection the second) so the UI bar keeps moving during
    a long scan; ``cancel`` is polled while sampling and detecting and
    aborts the scan with ``RuntimeError("face tracking cancelled")``.
    """
    import cv2
    import mediapipe as mp

    info = probe(video_path)
    frame_w, frame_h = info.width, info.height
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"cannot probe video size for {video_path}")
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValueError("clip end must be after start")

    # Crop size in native pixels for the target aspect (even, for yuv420p).
    aw, ah = aspect
    crop_w = int(frame_h * aw / ah)
    crop_h = frame_h
    if crop_w > frame_w:
        crop_w = frame_w
        crop_h = int(frame_w * ah / aw)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    # Sample timestamps (clip-relative) and frames. Frames are kept at
    # detection resolution only (see DETECT_MAX_WIDTH) — nothing downstream
    # needs their pixels, only the normalised boxes they produce.
    n_samples = max(1, int(round(duration * sample_fps)))
    times = [i / sample_fps for i in range(n_samples)]
    det_w, det_h = frame_w, frame_h
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        frames: list[np.ndarray | None] = []
        for i in range(n_samples):
            if cancel is not None and cancel():
                raise RuntimeError("face tracking cancelled")
            if progress is not None:
                progress(0.5 * (i + 1) / max(n_samples, 1))
            ok, frame = cap.read()
            if not ok or frame is None:
                frames.append(None)
                continue
            if frame.shape[1] > DETECT_MAX_WIDTH:
                scale = DETECT_MAX_WIDTH / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (DETECT_MAX_WIDTH, max(2, int(frame.shape[0] * scale))),
                )
            det_w, det_h = frame.shape[1], frame.shape[0]
            frames.append(frame)
            # Step to the next sample time.
            if i + 1 < n_samples:
                cap.set(cv2.CAP_PROP_POS_MSEC, (start + times[i + 1]) * 1000.0)
    finally:
        cap.release()

    # Audio activity (mono 16 kHz WAV) aligned to the sample times. Only
    # this range's audio is extracted — the old code decoded the WHOLE
    # source file per clip just for these RMS windows.
    audio_rms = [0.0] * n_samples
    with tempfile.TemporaryDirectory(prefix="clipforge-audio-") as tmp:
        wav_path = str(Path(tmp) / "audio.wav")
        try:
            extract_audio(
                video_path, wav_path, sample_rate=16000,
                start=start, duration=duration,
            )
            # The WAV starts at the range start; windows use clip time.
            audio_rms = _audio_rms_windows(wav_path, list(times))
        except RuntimeError:
            audio_rms = [0.0] * n_samples
    audio_peak = max(audio_rms) if audio_rms else 0.0
    audio_norm = [
        min(1.0, rms / (0.5 * audio_peak)) if audio_peak > 0 else 0.0
        for rms in audio_rms
    ]

    # Scene cuts via PySceneDetect; when it finds nothing (or is
    # unavailable) _detect_faces_in_frames falls back to frame-diff.
    # A false-positive cut only resets the speaker hold, so the more
    # sensitive fallback is safe here.
    cut_times = detect_scene_cuts(video_path, start=start, end=end)
    cut_flags = (
        scene_cut_flags(times, cut_times, start=start) if cut_times else None
    )

    # Detect faces + lip landmarks per sample (MediaPipe Tasks API).
    # Detection dimensions are the (possibly downscaled) frame size;
    # the returned boxes are normalised, so crop targets below still
    # multiply by the native frame size.
    detected, scene_cut = _detect_faces_in_frames(
        frames, det_w, det_h,
        min_detection_confidence=min_detection_confidence,
        scene_cuts=cut_flags,
        progress=(
            (lambda frac: progress(0.5 + 0.5 * frac))
            if progress is not None else None
        ),
        cancel=cancel,
    )

    # Lip movement: std of openness over a +-0.3 s window, per identity.
    lip_hist: dict[int, list[tuple[float, float]]] = {}
    for t, boxes in zip(times, detected):
        for box in boxes:
            lip_hist.setdefault(box.identity, []).append((t, box.lip_open))
    for t, boxes in zip(times, detected):
        for box in boxes:
            hist = lip_hist.get(box.identity, [])
            window = [v for tt, v in hist if abs(tt - t) <= 0.3]
            lip_std = float(np.std(window)) if len(window) >= 2 else 0.0
            box.speaking_score = min(1.0, lip_std / 0.03)
    # Audio gate applied per-sample (all faces share the clip's audio).
    for i, boxes in enumerate(detected):
        gate = 0.5 * audio_norm[i] if audio_peak > 0 else 0.0
        for box in boxes:
            box.speaking_score = box.speaking_score + gate

    # Active-speaker selection with a >= 1.5 s hold; scene cuts reset it.
    active: list[FaceBox | None] = []
    current_id: int | None = None
    hold_since = 0.0
    for i, boxes in enumerate(detected):
        t = times[i]
        if scene_cut[i]:
            current_id = None  # reset at scene cuts
        if not boxes:
            active.append(None)
            continue
        best = max(boxes, key=lambda b: b.speaking_score)
        if current_id is None:
            current_id = best.identity
            hold_since = t
        elif best.identity != current_id:
            # Switch only after the hold AND a clear lead.
            current_best = next(
                (b for b in boxes if b.identity == current_id), None
            )
            lead = best.speaking_score - (
                current_best.speaking_score if current_best else 0.0
            )
            if t - hold_since >= MIN_SPEAKER_HOLD_S and lead > 0.2:
                current_id = best.identity
                hold_since = t
        chosen = next((b for b in boxes if b.identity == current_id), best)
        active.append(chosen)

    # Multi-person split screen (Settings > Experimental): when two
    # people genuinely share the clip, each gets a face-framed pane
    # instead of one tracked crop jumping between them. A split that
    # fails its QA falls through to the single-speaker path below.
    if split_screen:
        pair = dominant_identity_pair(detected)
        if pair is not None:
            split = _split_crop_path(
                pair, detected, times, aspect, frame_w, frame_h
            )
            if split is not None:
                return split

    # Target crop per sample (native pixels), with headroom.
    fallback = sum(1 for face in active if face is None)
    target_x, target_y = _crop_targets(active, crop_w, crop_h, frame_w, frame_h)

    fallback_fraction = fallback / max(len(active), 1)
    if fallback_fraction >= 1.0:
        # No face anywhere: CENTRED FILL CROP. The crop window has the
        # output aspect and the renderer scales it to fill the whole
        # frame, so the clip still fills the 9:16 canvas edge to edge.
        # (The old "blur-fit" fallback fitted the whole 16:9 picture
        # inside the canvas over a blurred background — letterboxed
        # output; removed in v0.2.1.) Flagged needs_review so the Edit
        # step tells the user no face was found and framing is centred.
        cx = (frame_w - crop_w) / 2
        cy = (frame_h - crop_h) / 2
        return CropPath(
            times=list(times),
            boxes=[FaceBox(x=cx, y=cy, width=float(crop_w), height=float(crop_h))
                   for _ in times],
            faces=[None] * len(times),
            mode="track",
            qa_face_inside_fraction=1.0,
            qa_passed=True,
            needs_review=True,
            fallback_fraction=1.0,
        )

    def _build(lam: float, max_speed: float, project: bool) -> CropPath:
        xs = smooth_path_1d(target_x, times, lam=lam, max_speed=max_speed)
        ys = smooth_path_1d(target_y, times, lam=lam, max_speed=max_speed)
        if project:
            xs, ys = _project_to_contain(
                xs, ys, crop_w, crop_h, frame_w, frame_h, active
            )
        boxes = [
            FaceBox(x=x, y=y, width=float(crop_w), height=float(crop_h))
            for x, y in zip(xs, ys)
        ]
        # QA in normalized coordinates.
        norm_crops = [
            FaceBox(x=b.x / frame_w, y=b.y / frame_h,
                    width=b.width / frame_w, height=b.height / frame_h)
            for b in boxes
        ]
        frac = face_inside_crop_fraction(active, norm_crops)
        return CropPath(
            times=list(times),
            boxes=boxes,
            faces=list(active),
            mode="track",
            qa_face_inside_fraction=frac,
            qa_passed=frac >= QA_THRESHOLD,
            needs_review=False,
            fallback_fraction=fallback_fraction,
        )

    path = _build(DEFAULT_LAMBDA, DEFAULT_MAX_SPEED_PX_S, project=False)
    if not path.qa_passed:
        # Stricter retry: follow the face more closely, project the crop
        # onto the face so it is contained, then re-check.
        retry = _build(DEFAULT_LAMBDA / 4, DEFAULT_MAX_SPEED_PX_S / 2.5, project=True)
        path = retry
        if not path.qa_passed:
            path.needs_review = True
    return path
