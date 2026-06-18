# Python 3.10+
import socket
import json
import time
import threading
import os
import numpy as np
from typing import Dict, Any, Optional, List

HOST = "127.0.0.1"
PORT = 54322
SEND_HZ = 5.0  # 5Hz -> every 0.2s
TCP_NODELAY = True
os.environ['DEEPFACE_HOME'] = r'E:\Automatic\A_Python_Project\deepface'

class EmotionState:
    """Thread-safe latest emotion state storage."""
    def __init__(self):
        self._lock = threading.Lock()
        self._state: Optional[Dict[str, Any]] = None

    def set(self, state: Dict[str, Any]) -> None:
        with self._lock:
            self._state = state

    def get(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._state


# =========================
# DeepFace -> UE payload
# =========================

def _normalize_scores(scores: dict) -> dict:
    """
    DeepFace emotion 可能是 0-100（百分比）或 0-1。
    这里统一归一化到 sum=1，便于 UE 端处理。
    """
    if not scores:
        return {}
    vals = {k: float(v) for k, v in scores.items()}
    s = sum(vals.values())
    if s <= 0:
        return vals
    return {k: v / s for k, v in vals.items()}


def _region_to_bbox(region) -> dict:
    """
    你存的 region = faces_coordinates[idx]
    常见是 dict(x,y,w,h)。这里做容错：tuple/list 也支持。
    """
    if region is None:
        return {"x": 0, "y": 0, "w": 0, "h": 0}

    if isinstance(region, dict):
        return {
            "x": int(region.get("x", 0)),
            "y": int(region.get("y", 0)),
            "w": int(region.get("w", 0)),
            "h": int(region.get("h", 0)),
        }

    if isinstance(region, (list, tuple)) and len(region) >= 4:
        x, y, w, h = region[:4]
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    return {"x": 0, "y": 0, "w": 0, "h": 0}


def frame_data_to_payload(frame_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert your collected frame_data (with faces_info) into a stable payload for UE.
    frame_data 结构来自你改过的 deepface analysis_with_data_collection:
      {
        'timestamp': ...,
        'frame_id': ...,
        'num_faces': ...,
        'faces_info': [
            {
              'face_index': ...,
              'emotion': {...},
              'dominant_emotion': ...,
              'age': ...,
              'gender': {...},
              'race': {...},
              'region': {x,y,w,h} or tuple,
            }
        ]
      }
    """
    faces_payload = []
    for face in frame_data.get("faces_info", []):
        dominant = face.get("dominant_emotion", "unknown")
        scores = _normalize_scores(face.get("emotion", {}) or {})
        bbox = _region_to_bbox(face.get("region"))

        faces_payload.append({
            "face_index": int(face.get("face_index", 0)),
            "dominant": dominant,
            "scores": scores,
            "bbox": bbox,
            # 可选字段：UE 不用可删，减少数据量
            "age": face.get("age", None),
            "gender": face.get("gender", {}) or {},
            "race": face.get("race", {}) or {},
        })

    # 给 UE 一个默认的全局 dominant/scores（先取第一张脸）
    if faces_payload:
        global_dom = faces_payload[0]["dominant"]
        global_scores = faces_payload[0]["scores"]
    else:
        global_dom = "unknown"
        global_scores = {}

    return {
        "ts": float(frame_data.get("timestamp", time.time())),
        "frame_id": int(frame_data.get("frame_id", -1)),
        "num_faces": int(frame_data.get("num_faces", len(faces_payload))),
        "faces": faces_payload,
        "dominant": global_dom,
        "scores": global_scores,
    }

def json_safe(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if obj is None:
        return None

    # numpy scalar -> python scalar
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # containers
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    # fallback: keep python primitives, stringify others
    if isinstance(obj, (str, int, float, bool)):
        return obj

    return str(obj)

def start_deepface_stream(state_store: EmotionState, stop_event: threading.Event) -> None:
    """
    Run DeepFace stream in this process and push latest payload into state_store.

    注意：
    - DeepFace.stream_with_collection 是阻塞循环，所以必须在子线程跑
    - stop_event 目前无法优雅停止 DeepFace 内部循环（后续可以改源码支持）
    """
    from deepface import DeepFace

    def on_collected(data_list: List[Dict[str, Any]]) -> None:
        # 只取最新一帧：实时推送给UE
        if not data_list:
            return
        latest = data_list[-1]
        payload = frame_data_to_payload(latest)
        state_store.set(json_safe(payload))

    def on_frame(frame_data):
        payload = frame_data_to_payload(frame_data)
        state_store.set(json_safe(payload))

    # 让回调更频繁：1 秒一批（先跑通；想更实时可改 deepface 源码加 frame_callback）
    DeepFace.stream_with_collection(
        db_path="database",
        source=0,
        detector_backend="opencv",
        enable_face_analysis=True,
        time_threshold=1,
        frame_threshold=3,
        anti_spoofing=False,
        output_path=None,
        debug=False,
        collection_interval=1,
        data_callback=None,
        max_buffer_size=200,
        frame_callback=on_frame,
    )


# =========================
# TCP server
# =========================

def client_sender(conn: socket.socket, addr, state_store: EmotionState, stop_event: threading.Event) -> None:
    print(f"[server] client connected: {addr}")
    conn.settimeout(0.1)

    if TCP_NODELAY:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    send_interval = 1.0 / SEND_HZ
    last_sent_frame_id = -1

    try:
        while not stop_event.is_set():
            # Read any incoming bytes (optional)
            try:
                data = conn.recv(1024)
                if data:
                    # You can parse commands later
                    pass
            except socket.timeout:
                pass
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                break

            state = state_store.get()
            if state is not None:
                fid = int(state.get("frame_id", -1))
                if fid != last_sent_frame_id:
                    line = json.dumps(state, ensure_ascii=False) + "\n"
                    try:
                        conn.sendall(line.encode("utf-8"))
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                        break
                    last_sent_frame_id = fid

            time.sleep(send_interval)

    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[server] client disconnected: {addr}")


def run_server() -> None:
    state_store = EmotionState()
    stop_event = threading.Event()

    #  Start DeepFace stream thread (replaces fake inference loop)
    t_df = threading.Thread(target=start_deepface_stream, args=(state_store, stop_event), daemon=True)
    t_df.start()

    # Create server socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[server] listening on {HOST}:{PORT} (JSONL @ {SEND_HZ}Hz)")

    try:
        while True:
            conn, addr = srv.accept()
            # handle one client at a time (enough for UE local)
            client_sender(conn, addr, state_store, stop_event)
    except KeyboardInterrupt:
        print("\n[server] shutting down...")
    finally:
        stop_event.set()
        try:
            srv.close()
        except Exception:
            pass


if __name__ == "__main__":
    run_server()
