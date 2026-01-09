import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import inspireface as isf
import numpy as np
import onnxruntime as ort

VerifyResult = Dict[str, Union[str, float]]


class AletheiaEngine:
    """Face verification engine using InspireFace with batch support."""

    def __init__(
        self,
        classifier_path: str,
        debug_dir: str = "debug",
        quality_threshold: float = 0.45,
        max_faces: int = 1,
    ):
        self.classifier = ort.InferenceSession(classifier_path)
        self.face_session = self._init_inspireface(max_faces)
        self.quality_threshold = quality_threshold
        self.debug_dir = debug_dir
        self.similarity_threshold = isf.get_recommended_cosine_threshold()

        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)

    def _init_inspireface(self, max_faces: int) -> isf.InspireFaceSession:
        opt = isf.HF_ENABLE_FACE_RECOGNITION | isf.HF_ENABLE_QUALITY
        session = isf.InspireFaceSession(
            param=opt,
            detect_mode=isf.HF_DETECT_MODE_ALWAYS_DETECT,
            max_detect_num=max_faces,
            detect_pixel_level=160,
        )

        session.set_detection_confidence_threshold(0.5)
        return session

    def is_valid_document_classifier(self, img: np.ndarray) -> Tuple[bool, float]:
        """Validate document BEFORE face detection."""
        if img.size == 0:
            return False, 0.0

        img_resized = cv2.resize(img, (224, 224))
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_float - mean) / std
        blob = np.transpose(img_norm, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        outputs = self.classifier.run(
            None, {self.classifier.get_inputs()[0].name: blob}
        )
        logits = outputs[0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        confidence_doc = float(probs[0][1])

        return (confidence_doc > 0.80), confidence_doc

    def detect_and_extract(
        self, img: np.ndarray
    ) -> Tuple[Optional[object], Optional[object], Optional[Dict]]:
        if img is None or img.size == 0:
            return None, None, None

        faces = self.face_session.face_detection(img)

        if len(faces) == 0:
            return None, None, None

        if len(faces) > 1:
            largest_idx = max(
                range(len(faces)),
                key=lambda i: (faces[i].location[2] - faces[i].location[0])
                * (faces[i].location[3] - faces[i].location[1]),
            )
            face = faces[largest_idx]
        else:
            face = faces[0]

        x1, y1, x2, y2 = face.location
        bbox = [x1, y1, x2, y2]
        quality = (
            face.quality_confidence if hasattr(face, "quality_confidence") else 1.0
        )

        if quality < self.quality_threshold:
            return (
                None,
                face,
                {
                    "error": "low_quality",
                    "quality": quality,
                    "threshold": self.quality_threshold,
                    "bbox": bbox,
                },
            )

        feature = self.face_session.face_feature_extract(img, face)

        info = {
            "quality": quality,
            "bbox": bbox,
            "detection_confidence": face.detection_confidence,
            "angle": {"roll": face.roll, "pitch": face.pitch, "yaw": face.yaw},
        }

        return feature, face, info

    def _try_detect_with_rotation(
        self, img: np.ndarray
    ) -> Tuple[
        Optional[object], Optional[object], Optional[np.ndarray], Optional[Dict]
    ]:
        feature, face, info = self.detect_and_extract(img)
        if feature is not None:
            return feature, face, img, info

        for angle in [
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_180,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        ]:
            rotated = cv2.rotate(img.copy(), angle)
            feature, face, info = self.detect_and_extract(rotated)
            if feature is not None:
                return feature, face, rotated, info

        return None, None, None, None

    def verify_images(
        self, selfie_img: np.ndarray, document_img: np.ndarray
    ) -> VerifyResult:
        is_doc, conf = self.is_valid_document_classifier(document_img)
        if not is_doc:
            return {
                "status": "error",
                "message": f"Imagem do documento não reconhecida (Conf: {conf:.2f})",
            }

        feature_d, _, img_d_final, info_d = self._try_detect_with_rotation(document_img)

        if feature_d is None:
            error_msg = "Nenhum rosto encontrado no documento"
            if info_d and "error" in info_d:
                error_msg = f"Documento: {info_d['error']} (quality: {info_d.get('quality', 0):.2f})"
            return {"status": "error", "message": error_msg}

        feature_s, _, img_s_final, info_s = self._try_detect_with_rotation(selfie_img)

        if feature_s is None:
            error_msg = "Rosto não detectado na selfie"
            if info_s and "error" in info_s:
                error_msg = f"Selfie: {info_s['error']} (quality: {info_s.get('quality', 0):.2f})"
            return {"status": "error", "message": error_msg}

        similarity = isf.feature_comparison(feature_s, feature_d)
        status = "success" if similarity > self.similarity_threshold else "mismatch"

        bbox_s = info_s["bbox"] if info_s else None
        bbox_d = info_d["bbox"] if info_d else None
        self.generate_comparison_debug(
            img_s_final, bbox_s, img_d_final, bbox_d, similarity, status
        )

        percentage = isf.cosine_similarity_convert_to_percentage(similarity)

        return {
            "status": status,
            "similarity_score": similarity,
            "similarity_percentage": percentage,
            "threshold": self.similarity_threshold,
            "quality_selfie": info_s.get("quality", 0) if info_s else 0,
            "quality_document": info_d.get("quality", 0) if info_d else 0,
            "detection_confidence_selfie": (
                info_s.get("detection_confidence", 0) if info_s else 0
            ),
            "detection_confidence_document": (
                info_d.get("detection_confidence", 0) if info_d else 0
            ),
        }

    def verify_batch(
        self, selfie_images: List[np.ndarray], document_images: List[np.ndarray]
    ) -> List[VerifyResult]:
        """
        Batch verification for GPU optimization.

        Args:
            selfie_images: List of selfie images
            document_images: List of document images (must match length)

        Returns:
            List of VerifyResult dicts
        """
        assert len(selfie_images) == len(document_images), "Lists must have same length"

        results = []
        for selfie, document in zip(selfie_images, document_images):
            result = self.verify_images(selfie, document)
            results.append(result)

        return results

    def verify(self, selfie_path: str, document_path: str) -> VerifyResult:
        img_doc = cv2.imread(document_path)
        img_selfie = cv2.imread(selfie_path)

        if img_doc is None or img_selfie is None:
            return {"status": "error", "message": "Erro ao carregar arquivos de imagem"}

        return self.verify_images(img_selfie, img_doc)

    def generate_comparison_debug(self, img_s, bbox_s, img_d, bbox_d, score, status):
        if img_s is None or img_d is None:
            return

        c_s, c_d = img_s.copy(), img_d.copy()

        if bbox_s:
            cv2.rectangle(
                c_s, (bbox_s[0], bbox_s[1]), (bbox_s[2], bbox_s[3]), (0, 255, 0), 3
            )
        if bbox_d:
            cv2.rectangle(
                c_d, (bbox_d[0], bbox_d[1]), (bbox_d[2], bbox_d[3]), (255, 0, 0), 3
            )

        target_h = 600
        rs = target_h / c_s.shape[0]
        rd = target_h / c_d.shape[0]
        s_res = cv2.resize(c_s, (int(c_s.shape[1] * rs), target_h))
        d_res = cv2.resize(c_d, (int(c_d.shape[1] * rd), target_h))
        combined = np.hstack((s_res, d_res))

        color = (0, 255, 0) if status == "success" else (0, 0, 255)
        cv2.putText(
            combined,
            f"SCORE: {score:.4f} ({isf.cosine_similarity_convert_to_percentage(score):.1f}%) - {status.upper()}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
        cv2.imwrite(f"{self.debug_dir}/last_result.jpg", combined)

    def extract_feature(self, img: np.ndarray) -> Optional[object]:
        feature, _, _ = self.detect_and_extract(img)
        return feature
