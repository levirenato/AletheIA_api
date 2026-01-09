import redis
import uuid
from app.config import settings

r = redis.Redis(
    host=settings.redis_host, port=settings.redis_port, decode_responses=False
)


async def enqueue_verification(user_id: str, selfie_bytes: bytes, doc_bytes: bytes):
    job_id = str(uuid.uuid4())

    r.setex(f"img:selfie:{job_id}", settings.image_ttl, selfie_bytes)
    r.setex(f"img:doc:{job_id}", settings.image_ttl, doc_bytes)

    import json

    payload = json.dumps({"job_id": job_id, "user_id": user_id})
    r.rpush(settings.redis_queue_name, payload)

    return job_id
