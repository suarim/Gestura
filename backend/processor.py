import base64
import io
import os
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from pathlib import Path


class SignLanguageProcessor:
    """Utility class that mirrors the logic in capture_sign_2.py but operates on in-memory frames
    that come from the React front-end instead of capturing directly from the webcam.
    A session corresponds to a recording between the user hitting *Start* and *Stop*."""

    ROWS_PER_FRAME = 543  # Same constant as in the notebook

    def __init__(self,
                 xyz_skeleton_path: str | None = None,
                 tflite_model_path: str | None = None,
                 train_csv_path: str | None = None) -> None:
        # Resolve default paths relative to this file so that the service can be
        # launched from any working directory.
        base_dir = Path(__file__).resolve().parent

        if xyz_skeleton_path is None:
            xyz_skeleton_path = base_dir / "static" / "1460359.parquet"
        else:
            xyz_skeleton_path = Path(xyz_skeleton_path)

        if tflite_model_path is None:
            tflite_model_path = base_dir / "new_model.tflite"
        else:
            tflite_model_path = Path(tflite_model_path)

        if train_csv_path is None:
            train_csv_path = base_dir / "static" / "train.csv"
        else:
            train_csv_path = Path(train_csv_path)

        # Load constant skeleton required for padding missing landmarks
        if not xyz_skeleton_path.exists():
            raise FileNotFoundError(
                f"Skeleton parquet not found: {xyz_skeleton_path}. "
                "Ensure you have placed the file under backend/static/ or pass an explicit path."
            )
        self.xyz = pd.read_parquet(xyz_skeleton_path)

        # Initialise Mediapipe Holistic once for the entire process
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5)

        # Load TFLite model
        if not tflite_model_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_model_path}")
        self.interpreter = tf.lite.Interpreter(str(tflite_model_path))
        self.interpreter.allocate_tensors()
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")

        # Create mapping dicts from train.csv
        if not train_csv_path.exists():
            raise FileNotFoundError(f"train.csv not found: {train_csv_path}")
        train = pd.read_csv(train_csv_path)
        train["sign_ord"] = train["sign"].astype("category").cat.codes
        self.ORD2SIGN = (
            train[["sign_ord", "sign"]].set_index("sign_ord").squeeze().to_dict()
        )

        # Runtime session-level attributes
        self.reset_session()

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def reset_session(self) -> None:
        """Reset internal buffers before starting a new recording session."""
        self._landmarks_acc: List[pd.DataFrame] = []
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Core pipeline helpers (adapted from capture_sign_2.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _create_frame_landmarks_df(results, frame_idx: int, xyz_skel: pd.DataFrame) -> pd.DataFrame:
        mp_types = {"face": results.face_landmarks,
                    "pose": results.pose_landmarks,
                    "left_hand": results.left_hand_landmarks,
                    "right_hand": results.right_hand_landmarks}
        dfs = []
        for typ, landmark_obj in mp_types.items():
            if landmark_obj:
                pts = [[lm.x, lm.y, lm.z] for lm in landmark_obj.landmark]
                df = pd.DataFrame(pts, columns=["x", "y", "z"])
                df = df.reset_index().rename(columns={"index": "landmark_index"})
                df["type"] = typ
                dfs.append(df)
        if dfs:
            landmarks = pd.concat(dfs).reset_index(drop=True)
        else:
            # No landmarks detected in this frame – create empty placeholder rows
            landmarks = pd.DataFrame(columns=["landmark_index", "type", "x", "y", "z"])

        # Ensure we always have the full skeleton shape expected by the model
        xyz_skel = xyz_skel[["type", "landmark_index"]].drop_duplicates().reset_index(drop=True)
        landmarks = xyz_skel.merge(landmarks, on=["type", "landmark_index"], how="left")
        landmarks["frame"] = frame_idx
        return landmarks

    @staticmethod
    def _create_frame_windows(
        landmarks_df: pd.DataFrame, window_size: int = 20, overlap: int = 3
    ) -> List[np.ndarray]:
        """Split landmark dataframe into overlapping windows (ported as-is)."""
        frames = np.sort(landmarks_df["frame"].unique())
        stride = window_size - overlap
        windows: List[np.ndarray] = []
        data_cols = ["x", "y", "z"]
        for start in range(0, len(frames) - window_size + 1, stride):
            frame_slice = frames[start : start + window_size]
            w_data = landmarks_df[landmarks_df["frame"].isin(frame_slice)]
            if len(frame_slice) != window_size:
                continue
            window_arr: List[np.ndarray] = []
            for f in frame_slice:
                frame_vals = (
                    w_data[w_data["frame"] == f][data_cols].values.flatten()
                )
                window_arr.append(frame_vals)
            window_np = np.array(window_arr).reshape(
                window_size, SignLanguageProcessor.ROWS_PER_FRAME, len(data_cols)
            )
            windows.append(window_np.astype(np.float32))
        return windows

    # ------------------------------------------------------------------
    # Public API used from FastAPI routes
    # ------------------------------------------------------------------
    def add_frame(self, b64_img: str) -> None:
        """Decode base64 image string, run Mediapipe and save landmarks."""
        # Remove header if present e.g. "data:image/jpeg;base64,"
        if "," in b64_img:
            b64_img = b64_img.split(",", 1)[1]
        try:
            img_bytes = base64.b64decode(b64_img)
        except base64.binascii.Error as e:
            raise ValueError("Invalid base64 image data") from e

        # Convert bytes to numpy BGR image for cv2 -> then to RGB for Mediapipe
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Could not decode image")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        self._frame_idx += 1
        results = self.holistic.process(frame_rgb)
        landmarks_df = self._create_frame_landmarks_df(results, self._frame_idx, self.xyz)
        self._landmarks_acc.append(landmarks_df)

    def translate(self) -> str:
        """Run the accumulated frames through the TFLite model and (optionally) Ollama.
        Returns the final predicted sentence."""
        if not self._landmarks_acc:
            return "No frames received."

        combined = pd.concat(self._landmarks_acc).reset_index(drop=True)
        windows = self._create_frame_windows(combined, window_size=5, overlap=3)
        if not windows:
            return "Insufficient data for prediction."

        window_predictions = []
        for idx, window in enumerate(windows):
            prediction = self.prediction_fn(inputs=window)
            outputs = prediction["outputs"]
            sign_idx = int(outputs.argmax())
            confidence = float(outputs.max())
            sign = self.ORD2SIGN.get(sign_idx)
            if sign is None:
                continue
            # Simple filtering rules (same as capture_sign_2)
            if confidence < 0.15 or (window_predictions and window_predictions[-1][0] == sign):
                continue
            window_predictions.append((sign, confidence))

        if not window_predictions:
            return "No confident prediction."  # Fallback message

        # Majority vote
        counts = {}
        for sign, _ in window_predictions:
            counts[sign] = counts.get(sign, 0) + 1
        most_common_sign = max(counts.items(), key=lambda kv: kv[1])[0]

        # Optional: use Ollama for sentence generation
        final_signs = [sign for sign, _ in window_predictions]
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        prompt = (
            "Translate the following sequence of sign language words into a simple "
            "English phrase. Use only the given words and the absolute minimum necessary "
            "connecting words (like 'on', 'in', 'at'). Signs: "
            + ", ".join(final_signs)
            + ". Just give final sentence, no explanation or additional text. "
            + "And if ever tv pops up replace it with apple."
        )

        try:
            resp = requests.post(
                ollama_url, json={"model": ollama_model, "prompt": prompt, "stream": False}, timeout=30
            )
            resp.raise_for_status()
            generated_sentence = resp.json().get("response", "").strip()
            if generated_sentence:
                return generated_sentence
        except Exception:
            # Fall back to just returning the most common sign if Ollama fails
            pass
        return most_common_sign 