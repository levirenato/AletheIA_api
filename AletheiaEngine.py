import os
from typing import Dict, List, Optional, Tuple, Union, cast

import cv2
import numpy as np
import onnxruntime as ort

VerifyResult = Dict[str, Union[str, float]]


class AletheiaEngine:
    def __init__(
        self,
        ultraface_path: str,
        arcface_path: str,
        classifier_path: str,
        leann_path: str = "data/leann_storage.npy",
        debug_dir: str = "debug",
    ):
        self.detector = ort.InferenceSession(ultraface_path)
        self.extractor = ort.InferenceSession(arcface_path)
        self.classifier = ort.InferenceSession(classifier_path)

        self.leann_path = leann_path
        self.debug_dir = debug_dir

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

        self.vectors: np.ndarray
        self.ids: List[str]
        self.vectors, self.ids = self._load_leann()

    def _load_leann(self) -> Tuple[np.ndarray, List[str]]:
        if os.path.exists(self.leann_path):
            data = np.load(self.leann_path, allow_pickle=True).item()
            return data["vectors"], data["ids"]
        return np.empty((0, 512), dtype=np.float32), []

    def _save_leann(self) -> None:
        os.makedirs(os.path.dirname(self.leann_path), exist_ok=True)
        payload = {"vectors": self.vectors, "ids": self.ids}
        np.save(self.leann_path, np.array(payload))

    def is_valid_document_classifier(self, img: np.ndarray) -> Tuple[bool, float]:
        """Usa o seu MobileNetV3 para validar se a imagem é um documento."""
        if img.size == 0:
            return False, 0.0
        img_resized = cv2.resize(img, (224, 224))
        img_float = img_resized.astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_norm = (img_float - mean) / std
        blob = np.transpose(img_norm, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        outputs = self.classifier.run(
            None, {self.classifier.get_inputs()[0].name: blob}
        )
        logits = cast(np.ndarray, outputs[0])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        confidence_doc = float(probs[0][1])
        return (confidence_doc > 0.80), confidence_doc

    def _detect_faces_raw(
        self, img: np.ndarray
    ) -> Tuple[List[np.ndarray], List[List[int]]]:
        """Detecta rostos garantindo que detecções duplicadas do mesmo rosto sejam filtradas."""
        if img is None or img.size == 0:
            return [], []
        h, w = img.shape[:2]

        blob_img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
        blob = (blob_img.astype(np.float32) - 127.0) / 128.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        outputs = self.detector.run(None, {self.detector.get_inputs()[0].name: blob})
        confidences, boxes = cast(np.ndarray, outputs[0]), cast(np.ndarray, outputs[1])

        scores = confidences[0, :, 1]
        mask = scores > 0.95

        raw_faces, raw_boxes = [], []
        for idx in np.where(mask)[0]:
            box = boxes[0, idx]
            x1, y1, x2, y2 = (
                int(box[0] * w),
                int(box[1] * h),
                int(box[2] * w),
                int(box[3] * h),
            )
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if (x2 - x1) < (w * 0.05) or (y2 - y1) < (h * 0.05):
                continue

            raw_faces.append(img[y1:y2, x1:x2])
            raw_boxes.append([x1, y1, x2, y2])

        if len(raw_boxes) > 1:
            return self._filter_overlapping_boxes(raw_faces, raw_boxes)

        return raw_faces, raw_boxes

    def _filter_overlapping_boxes(self, faces, boxes):
        """Se houver boxes muito próximos, assume que é o mesmo rosto."""
        if not boxes:
            return [], []

        final_faces, final_boxes = [faces[0]], [boxes[0]]

        for i in range(1, len(boxes)):
            b1, b2 = final_boxes[-1], boxes[i]
            overlap = not (
                b2[0] > b1[2] or b2[2] < b1[0] or b2[1] > b1[3] or b2[3] < b1[1]
            )
            if not overlap:
                final_faces.append(faces[i])
                final_boxes.append(boxes[i])

        return final_faces, final_boxes

    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        face_img_resized = cv2.resize(face_img, (112, 112))
        face_img_resized = (face_img_resized.astype(np.float32) - 127.5) / 128.0
        face_img_resized = np.expand_dims(face_img_resized, axis=0)
        outputs = self.extractor.run(
            None, {self.extractor.get_inputs()[0].name: face_img_resized}
        )
        embedding = cast(np.ndarray, outputs[0]).flatten()
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 1e-6 else embedding

    def check_leann_duplicity(
        self, emb: np.ndarray, threshold: float = 0.6
    ) -> Tuple[bool, Optional[str]]:
        if self.vectors.shape[0] == 0:
            return False, None
        sims = np.dot(self.vectors, emb)
        idx = int(np.argmax(sims))
        if sims[idx] > threshold:
            return True, self.ids[idx]
        return False, None

    def verify(
        self, selfie_path: str, document_path: str, user_id: str
    ) -> VerifyResult:
        img_doc = cv2.imread(document_path)
        img_selfie = cv2.imread(selfie_path)
        if img_doc is None or img_selfie is None:
            return {"status": "error", "message": "Erro ao carregar arquivos de imagem"}

        is_doc, conf = self.is_valid_document_classifier(img_doc)
        if not is_doc:
            return {
                "status": "error",
                "message": f"Imagem do documento não reconhecida (Conf: {conf:.2f})",
            }

        face_d, box_d, img_d_final = None, None, None
        for angle in [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]:
            current = img_doc.copy() if angle is None else cv2.rotate(img_doc, angle)
            faces, boxes = self._detect_faces_raw(current)

            if len(faces) == 1:
                face_d, box_d, img_d_final = faces[0], boxes[0], current
                break
            elif len(faces) > 1:
                return {
                    "status": "error",
                    "message": "Detectado mais de um rosto no documento",
                }

        if face_d is None:
            return {
                "status": "error",
                "message": "Nenhum rosto encontrado no documento",
            }

        emb_d = self.get_embedding(face_d)
        is_dup, old_id = self.check_leann_duplicity(emb_d)
        if is_dup:
            return {
                "status": "fraud",
                "message": f"Documento já cadastrado (ID: {old_id})",
            }

        face_s, box_s, img_s_final = None, None, None
        for angle in [
            None,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]:
            current = (
                img_selfie.copy() if angle is None else cv2.rotate(img_selfie, angle)
            )
            faces, boxes = self._detect_faces_raw(current)

            if len(faces) == 1:
                face_s, box_s, img_s_final = faces[0], boxes[0], current
                break

        if face_s is None:
            return {"status": "error", "message": "Rosto não detectado na selfie"}

        emb_s = self.get_embedding(face_s)
        similarity = float(np.dot(emb_s, emb_d))
        status = "success" if similarity > 0.80 else "mismatch"

        self.generate_comparison_debug(
            img_s_final, box_s, img_d_final, box_d, similarity, status
        )

        if status == "success":
            self.vectors = np.vstack([self.vectors, emb_d])
            self.ids.append(user_id)
            self._save_leann()

        return {"status": status, "score": similarity}

    def generate_comparison_debug(self, img_s, box_s, img_d, box_d, score, status):
        """Gera imagem de debug lado a lado."""
        c_s, c_d = img_s.copy(), img_d.copy()
        cv2.rectangle(c_s, (box_s[0], box_s[1]), (box_s[2], box_s[3]), (0, 255, 0), 3)
        cv2.rectangle(c_d, (box_d[0], box_d[1]), (box_d[2], box_d[3]), (255, 0, 0), 3)

        target_h = 600
        rs, rd = target_h / c_s.shape[0], target_h / c_d.shape[0]
        s_res = cv2.resize(c_s, (int(c_s.shape[1] * rs), target_h))
        d_res = cv2.resize(c_d, (int(c_d.shape[1] * rd), target_h))

        combined = np.hstack((s_res, d_res))
        cv2.putText(
            combined,
            f"SCORE: {score:.4f} - {status.upper()}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0) if status == "success" else (0, 0, 255),
            3,
        )

        cv2.imwrite(f"{self.debug_dir}/last_result.jpg", combined)
