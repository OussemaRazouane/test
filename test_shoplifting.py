"""
test_shoplifting.py -- standalone, single-file live test for the shoplifting
pipeline (CNN classifier + pose confirmation gate) against an RTSP stream.

Self-contained: no imports from the ai-engine project, no database, no
InfluxDB. Only needs onnxruntime, opencv-python, numpy, python-dotenv.

Config comes from a .env file placed next to this script. See the bottom of
this file for the exact keys it reads.

Press 'q' in the video window to quit.
"""

import os
import time
import logging
from dotenv import load_dotenv
import cv2
import numpy as np
import onnxruntime as ort

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("shoplifting_test")

# ── Config from .env ──────────────────────────────────────────────
RTSP_URL = os.getenv("RTSP_URL", "0")  # falls back to webcam 0 if unset

PERSON_MODEL      = os.getenv("PERSON_MODEL", "../ai-engine/models/person.onnx")
SHOPLIFTING_MODEL = os.getenv("SHOPLIFTING_MODEL", "../ai-engine/models/shoplifting.onnx")
POSE_MODEL        = os.getenv("POSE_MODEL", "../ai-engine/models/pose.onnx")

PERSON_CONF      = float(os.getenv("PERSON_CONF", "0.7"))
SHOPLIFTING_CONF = float(os.getenv("SHOPLIFTING_CONF", "0.5"))
POSE_CONF        = float(os.getenv("POSE_CONF", "0.5"))
NMS_THRESHOLD    = float(os.getenv("NMS_THRESHOLD", "0.3"))

CONFIRM_WINDOW_SECONDS    = float(os.getenv("SHOPLIFT_CONFIRM_WINDOW_SECONDS", "2.5"))
CONCEAL_ZONE_RADIUS_RATIO = float(os.getenv("SHOPLIFT_CONCEAL_ZONE_RADIUS_RATIO", "0.35"))
CONCEAL_DWELL_FRAMES      = int(os.getenv("SHOPLIFT_CONCEAL_DWELL_FRAMES", "5"))

FRAME_WIDTH  = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))

# ── TensorRT config (mirrors production ai-engine .env keys) ──────
TRT_CACHE_DIR      = os.getenv("TRT_CACHE_DIR", "./trt_cache")
TRT_MAX_WORKSPACE  = int(os.getenv("TRT_MAX_WORKSPACE", str(2 * 1024 * 1024 * 1024)))  # 2GB default
TRT_FP16           = os.getenv("TRT_FP16", "true").lower() == "true"
# Set to "false" in .env to force-skip TensorRT even if the provider is
# available (useful for isolating whether a bug is TRT-specific).
USE_TENSORRT       = os.getenv("USE_TENSORRT", "true").lower() == "true"

SHOPLIFTING_CLASS_ID = 0
PERSON_CLASS_ID = 0

# COCO pose keypoint indices (fixed by the pretrained pose model)
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW       = 7, 8
KP_L_WRIST, KP_R_WRIST       = 9, 10
KP_L_HIP, KP_R_HIP           = 11, 12
KP_L_KNEE, KP_R_KNEE         = 13, 14
KP_L_ANKLE, KP_R_ANKLE       = 15, 16

SKELETON_PAIRS = [
    (KP_L_SHOULDER, KP_R_SHOULDER), (KP_L_SHOULDER, KP_L_ELBOW), (KP_L_ELBOW, KP_L_WRIST),
    (KP_R_SHOULDER, KP_R_ELBOW), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP), (KP_R_SHOULDER, KP_R_HIP), (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE), (KP_L_KNEE, KP_L_ANKLE),
    (KP_R_HIP, KP_R_KNEE), (KP_R_KNEE, KP_R_ANKLE),
]

C_PERSON   = (0, 255, 0)
C_CNN_RAW  = (0, 255, 255)
C_CONFIRM  = (0, 0, 255)
C_SKELETON = (255, 255, 255)
C_KEYPOINT = (255, 255, 0)
C_ZONE     = (255, 0, 255)


def resolve_providers():
    """Build the ONNX Runtime provider chain: TensorRT -> CUDA -> CPU,
    matching the production ai-engine fallback chain exactly."""
    available = ort.get_available_providers()
    log.info(f"Available ONNX providers: {available}")

    if USE_TENSORRT and "TensorrtExecutionProvider" in available:
        os.makedirs(TRT_CACHE_DIR, exist_ok=True)
        log.info(
            f"Using TensorRT (fp16={TRT_FP16}, cache={TRT_CACHE_DIR}, "
            f"workspace={TRT_MAX_WORKSPACE / (1024**3):.1f}GB)"
        )
        trt_options = {
            "device_id": 0,
            "trt_fp16_enable": TRT_FP16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE_DIR,
            "trt_max_workspace_size": TRT_MAX_WORKSPACE,
        }
        return (
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            [trt_options, {"device_id": 0}, {}],
        )

    if not USE_TENSORRT:
        log.info("USE_TENSORRT=false -- skipping TensorRT even though it may be available")

    if "CUDAExecutionProvider" in available:
        log.info("Using CUDA (TensorRT not available/disabled)")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"], [{"device_id": 0}, {}]

    log.warning("No GPU provider found -- using CPU (slower, fine for tuning)")
    return ["CPUExecutionProvider"], [{}]


class YoloDetector:
    """Minimal YOLOv8 ONNX detector -- works for both person.onnx (80 COCO
    classes) and shoplifting.onnx (2 classes), same output shape family
    (1, C+4, N), just a different class_filter."""

    def __init__(self, onnx_path, conf_thresh, class_filter, providers, provider_options):
        self.conf_thresh = conf_thresh
        self.class_filter = class_filter
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model not found: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=providers, provider_options=provider_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = self.session.get_inputs()[0].shape[2]
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        log.info(f"Warming up {os.path.basename(onnx_path)} (TRT engine build happens here if applicable)...")
        warmup_start = time.monotonic()
        self.session.run(self.output_names, {self.input_name: dummy})
        log.info(
            f"[{self.session.get_providers()[0]}] {os.path.basename(onnx_path)} ready "
            f"(warmup took {time.monotonic() - warmup_start:.1f}s)"
        )

    def detect(self, frame):
        tensor = self._preprocess(frame)
        raw = self.session.run(self.output_names, {self.input_name: tensor})
        return self._postprocess(raw[0], frame.shape)

    def _preprocess(self, img):
        resized = cv2.resize(img, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normed, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _postprocess(self, output, orig_shape):
        if output.ndim == 3:
            output = output[0]
        output = output.T
        h_orig, w_orig = orig_shape[:2]
        sx, sy = w_orig / self.input_size, h_orig / self.input_size
        boxes = []
        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if conf < self.conf_thresh:
                continue
            if self.class_filter is not None and class_id not in self.class_filter:
                continue
            cx, cy, w, h = row[:4]
            x1 = int(max(0, (cx - w / 2) * sx))
            y1 = int(max(0, (cy - h / 2) * sy))
            x2 = int(min(w_orig, (cx + w / 2) * sx))
            y2 = int(min(h_orig, (cy + h / 2) * sy))
            boxes.append((x1, y1, x2, y2, conf, class_id))
        return self._nms(boxes)

    def _nms(self, boxes):
        if not boxes:
            return []
        cv2_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        scores = [b[4] for b in boxes]
        indices = cv2.dnn.NMSBoxes(cv2_boxes, scores, score_threshold=0.0, nms_threshold=NMS_THRESHOLD)
        if len(indices) == 0:
            return []
        return [boxes[i] for i in (indices.flatten() if hasattr(indices, "flatten") else indices)]


class PoseEstimator:
    """YOLOv8-pose ONNX wrapper. Output shape (1, 56, 8400):
    4 box coords + 1 person confidence + 17 keypoints x (x, y, visibility)."""

    def __init__(self, onnx_path, conf_thresh, providers, provider_options):
        self.conf_thresh = conf_thresh
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Model not found: {onnx_path}")
        self.session = ort.InferenceSession(onnx_path, providers=providers, provider_options=provider_options)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = self.session.get_inputs()[0].shape[2]
        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
        log.info(f"Warming up {os.path.basename(onnx_path)} (TRT engine build happens here if applicable)...")
        warmup_start = time.monotonic()
        self.session.run(self.output_names, {self.input_name: dummy})
        log.info(
            f"[{self.session.get_providers()[0]}] {os.path.basename(onnx_path)} ready "
            f"(warmup took {time.monotonic() - warmup_start:.1f}s)"
        )

    def detect(self, frame):
        tensor = self._preprocess(frame)
        raw = self.session.run(self.output_names, {self.input_name: tensor})
        return self._postprocess(raw[0], frame.shape)

    def _preprocess(self, img):
        resized = cv2.resize(img, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normed = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normed, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def _postprocess(self, output, orig_shape):
        if output.ndim == 3:
            output = output[0]
        output = output.T
        h_orig, w_orig = orig_shape[:2]
        sx, sy = w_orig / self.input_size, h_orig / self.input_size
        candidates = []
        for row in output:
            conf = float(row[4])
            if conf < self.conf_thresh:
                continue
            cx, cy, w, h = row[:4]
            x1 = max(0.0, (cx - w / 2) * sx)
            y1 = max(0.0, (cy - h / 2) * sy)
            x2 = min(float(w_orig), (cx + w / 2) * sx)
            y2 = min(float(h_orig), (cy + h / 2) * sy)
            kpts = row[5:].reshape(17, 3).copy()
            kpts[:, 0] *= sx
            kpts[:, 1] *= sy
            candidates.append((x1, y1, x2, y2, conf, kpts))
        return self._nms(candidates)

    def _nms(self, candidates):
        if not candidates:
            return []
        boxes = [[c[0], c[1], c[2] - c[0], c[3] - c[1]] for c in candidates]
        scores = [c[4] for c in candidates]
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=NMS_THRESHOLD)
        if len(indices) == 0:
            return []
        result = []
        for i in (indices.flatten() if hasattr(indices, "flatten") else indices):
            x1, y1, x2, y2, conf, kpts = candidates[i]
            result.append({"box": (int(x1), int(y1), int(x2), int(y2)), "conf": conf, "keypoints": kpts})
        return result


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    __slots__ = ("box", "last_seen", "window_open_until", "dwell_count")

    def __init__(self, box):
        self.box = box
        self.last_seen = time.monotonic()
        self.window_open_until = None
        self.dwell_count = 0


class ShopliftingConfirmer:
    """Pose-based confirmation gate -- identical reach-then-dwell logic to
    the production ai-engine/app/ai/shoplifting_confirmer.py, copied inline
    here to keep this a single self-contained file."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 0

    def has_open_windows(self):
        now = time.monotonic()
        return any(t.window_open_until is not None and t.window_open_until > now for t in self.tracks.values())

    def update(self, persons, cnn_boxes, poses):
        now = time.monotonic()
        matched = self._match_tracks(persons)

        for cbox in cnn_boxes:
            cx1, cy1, cx2, cy2 = cbox[:4]
            best_tid, best_iou = None, 0.1
            for tid, pbox in matched.items():
                i = iou((cx1, cy1, cx2, cy2), pbox)
                if i > best_iou:
                    best_iou, best_tid = i, tid
            if best_tid is not None:
                self.tracks[best_tid].window_open_until = now + CONFIRM_WINDOW_SECONDS
                self.tracks[best_tid].dwell_count = 0

        confirmed = []
        pose_by_box = {p["box"]: p for p in poses}
        for tid, pbox in matched.items():
            track = self.tracks[tid]
            if track.window_open_until is None or track.window_open_until <= now:
                continue
            pose = self._find_pose(pbox, pose_by_box)
            if pose is None:
                continue
            if self._wrist_in_zone(pose):
                track.dwell_count += 1
            else:
                track.dwell_count = 0
            if track.dwell_count >= CONCEAL_DWELL_FRAMES:
                x1, y1, x2, y2 = pbox
                confirmed.append((x1, y1, x2, y2, pose.get("conf", 0.9), 0))
                track.window_open_until = None
                track.dwell_count = 0

        self._prune(now)
        return confirmed

    def _match_tracks(self, persons):
        now = time.monotonic()
        boxes = [(p[0], p[1], p[2], p[3]) for p in persons]
        result, used = {}, set()
        for box in boxes:
            best_tid, best_iou = None, 0.3
            for tid, track in self.tracks.items():
                if tid in used:
                    continue
                i = iou(box, track.box)
                if i > best_iou:
                    best_iou, best_tid = i, tid
            if best_tid is not None:
                self.tracks[best_tid].box = box
                self.tracks[best_tid].last_seen = now
                result[best_tid] = box
                used.add(best_tid)
            else:
                nid = self.next_id
                self.next_id += 1
                self.tracks[nid] = Track(box)
                result[nid] = box
        return result

    def _prune(self, now):
        stale = [tid for tid, t in self.tracks.items() if now - t.last_seen > 3.0]
        for tid in stale:
            del self.tracks[tid]

    def _find_pose(self, pbox, pose_by_box):
        best_pose, best_iou = None, 0.3
        for pb, pose in pose_by_box.items():
            i = iou(pbox, pb)
            if i > best_iou:
                best_iou, best_pose = i, pose
        return best_pose

    def _wrist_in_zone(self, pose):
        kpts = pose["keypoints"]
        l_hip, r_hip = kpts[KP_L_HIP], kpts[KP_R_HIP]
        l_sh, r_sh = kpts[KP_L_SHOULDER], kpts[KP_R_SHOULDER]
        l_wr, r_wr = kpts[KP_L_WRIST], kpts[KP_R_WRIST]
        if l_hip[2] < 0.3 or r_hip[2] < 0.3 or l_sh[2] < 0.3 or r_sh[2] < 0.3:
            return False
        hip_cx, hip_cy = (l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2
        sh_cx, sh_cy = (l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2
        torso = ((sh_cx - hip_cx) ** 2 + (sh_cy - hip_cy) ** 2) ** 0.5
        if torso < 1e-3:
            return False
        zone_r = torso * CONCEAL_ZONE_RADIUS_RATIO
        for wrist in (l_wr, r_wr):
            if wrist[2] < 0.3:
                continue
            d = ((wrist[0] - hip_cx) ** 2 + (wrist[1] - hip_cy) ** 2) ** 0.5
            if d <= zone_r:
                return True
        return False


def draw_skeleton(frame, keypoints, vis_thresh=0.3):
    for a, b in SKELETON_PAIRS:
        if keypoints[a][2] < vis_thresh or keypoints[b][2] < vis_thresh:
            continue
        pa = (int(keypoints[a][0]), int(keypoints[a][1]))
        pb = (int(keypoints[b][0]), int(keypoints[b][1]))
        cv2.line(frame, pa, pb, C_SKELETON, 2)
    for (x, y, v) in keypoints:
        if v < vis_thresh:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, C_KEYPOINT, -1)


def draw_conceal_zone(frame, keypoints, vis_thresh=0.3):
    l_hip, r_hip = keypoints[KP_L_HIP], keypoints[KP_R_HIP]
    l_sh, r_sh = keypoints[KP_L_SHOULDER], keypoints[KP_R_SHOULDER]
    if l_hip[2] < vis_thresh or r_hip[2] < vis_thresh or l_sh[2] < vis_thresh or r_sh[2] < vis_thresh:
        return
    hip_cx, hip_cy = (l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2
    sh_cx, sh_cy = (l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2
    torso = ((sh_cx - hip_cx) ** 2 + (sh_cy - hip_cy) ** 2) ** 0.5
    radius = int(torso * CONCEAL_ZONE_RADIUS_RATIO)
    cv2.circle(frame, (int(hip_cx), int(hip_cy)), radius, C_ZONE, 2)
    cv2.circle(frame, (int(hip_cx), int(hip_cy)), 3, C_ZONE, -1)


def draw_hud(frame, fps, window_open, confirmed_count, active_provider):
    lines = [
        f"FPS: {fps:.1f}   (press 'q' to quit)",
        f"Provider: {active_provider}",
        f"Confirmation window open: {'YES' if window_open else 'no'}",
        f"CONFIRMED this run: {confirmed_count}",
        f"CONF={SHOPLIFTING_CONF} WINDOW={CONFIRM_WINDOW_SECONDS}s "
        f"ZONE={CONCEAL_ZONE_RADIUS_RATIO} DWELL={CONCEAL_DWELL_FRAMES}f",
    ]
    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def main():
    log.info(f"RTSP source: {RTSP_URL}")
    providers, provider_options = resolve_providers()

    person_detector = YoloDetector(PERSON_MODEL, PERSON_CONF, [PERSON_CLASS_ID], providers, provider_options)
    shoplifting_detector = YoloDetector(SHOPLIFTING_MODEL, SHOPLIFTING_CONF, [SHOPLIFTING_CLASS_ID], providers, provider_options)
    pose_estimator = PoseEstimator(POSE_MODEL, POSE_CONF, providers, provider_options)

    # Whichever provider actually ended up active after ORT's internal fallback
    # (e.g. TRT engine build can silently fall back to CUDA on failure).
    active_provider = person_detector.session.get_providers()[0]

    confirmer = ShopliftingConfirmer()

    source = int(RTSP_URL) if RTSP_URL.isdigit() else RTSP_URL
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Could not open stream: {RTSP_URL}")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    log.info("Stream opened. Press 'q' in the video window to quit.")

    confirmed_count = 0
    frame_count = 0
    fps = 0.0
    last_fps_time = time.monotonic()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("Read failed, retrying...")
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            annotated = frame.copy()

            persons = person_detector.detect(frame)
            cnn_boxes, poses, confirmed = [], [], []

            if persons:
                cnn_boxes = shoplifting_detector.detect(frame)
                if cnn_boxes or confirmer.has_open_windows():
                    poses = pose_estimator.detect(frame)
                confirmed = confirmer.update(persons, cnn_boxes, poses)
                confirmed_count += len(confirmed)

            for (x1, y1, x2, y2, conf, _) in persons:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), C_PERSON, 1)
            for (x1, y1, x2, y2, conf, _) in cnn_boxes:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), C_CNN_RAW, 2)
                cv2.putText(annotated, f"CNN raw {conf:.2f}", (x1, max(y1 - 8, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CNN_RAW, 2)
            for (x1, y1, x2, y2, conf, _) in confirmed:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), C_CONFIRM, 3)
                cv2.putText(annotated, "CONFIRMED SHOPLIFTING", (x1, max(y1 - 12, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CONFIRM, 2)
            for pose in poses:
                draw_skeleton(annotated, pose["keypoints"])
                draw_conceal_zone(annotated, pose["keypoints"])

            frame_count += 1
            now = time.monotonic()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            draw_hud(annotated, fps, confirmer.has_open_windows(), confirmed_count, active_provider)

            cv2.imshow("Shoplifting Pipeline Test", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.info(f"Total confirmed shoplifting events this run: {confirmed_count}")


if __name__ == "__main__":
    main()