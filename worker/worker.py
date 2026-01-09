import redis
import json
import numpy as np
import cv2
from AletheiaEngine import AletheiaEngine

engine = AletheiaEngine(classifier_path="models/mobilenet.onnx")
r = redis.Redis(host="redis", port=6379)


def run_worker():
    while True:
        _, task = r.blpop("face_tasks")
        data = json.loads(task)
        job_id = data["job_id"]

        img_s = cv2.imdecode(np.frombuffer(r.get(f"img:s:{job_id}"), np.uint8), 1)
        img_d = cv2.imdecode(np.frombuffer(r.get(f"img:d:{job_id}"), np.uint8), 1)

        result = engine.verify_images(img_s, img_d)

        feat = engine.extract_feature(img_s)
        if feat is not None:
            result["embedding"] = feat.tolist()

        r.setex(f"result:{job_id}", 300, json.dumps(result))
        r.publish(f"channel:{job_id}", "done")
