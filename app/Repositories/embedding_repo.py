from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Embedding


class EmbeddingRepository:
    """Repository for embedding operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_user_id(self, user_id: str) -> Optional[Embedding]:
        """Get embedding by user_id."""
        result = await self.session.execute(
            select(Embedding).where(Embedding.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_id(self, embedding_id: UUID) -> Optional[Embedding]:
        """Get embedding by id."""
        result = await self.session.execute(
            select(Embedding).where(Embedding.id == embedding_id)
        )
        return result.scalar_one_or_none()

    async def create(
        self, user_id: str, embedding: np.ndarray, metadata: dict | None = None
    ) -> Embedding:
        """Create new embedding."""
        db_embedding = Embedding(
            user_id=user_id,
            embedding=embedding.tolist(),
            meta=metadata or {},
        )
        self.session.add(db_embedding)
        await self.session.flush()
        await self.session.refresh(db_embedding)
        return db_embedding

    async def update(
        self, user_id: str, embedding: np.ndarray, metadata: dict | None = None
    ) -> Optional[Embedding]:
        """Update existing embedding."""
        db_embedding = await self.get_by_user_id(user_id)
        if not db_embedding:
            return None

        db_embedding.embedding = embedding.tolist()
        if metadata:
            db_embedding.meta = metadata

        await self.session.flush()
        await self.session.refresh(db_embedding)
        return db_embedding

    async def delete(self, user_id: str) -> bool:
        """Delete embedding by user_id."""
        db_embedding = await self.get_by_user_id(user_id)
        if not db_embedding:
            return False

        await self.session.delete(db_embedding)
        await self.session.flush()
        return True

    async def find_similar(
        self, embedding: np.ndarray, threshold: float = 0.6, limit: int = 5
    ) -> list[tuple[Embedding, float]]:
        """
        Find similar embeddings using cosine distance.

        Returns list of (Embedding, similarity_score) tuples.
        Similarity score: 0 = orthogonal, 1 = identical
        """
        embedding_list = embedding.tolist()

        query = (
            select(
                Embedding,
                (1 - Embedding.embedding.cosine_distance(embedding_list)).label(
                    "similarity"
                ),
            )
            .order_by(Embedding.embedding.cosine_distance(embedding_list))
            .limit(limit)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return [
            (row.Embedding, row.similarity)
            for row in rows
            if row.similarity >= threshold
        ]

    async def check_duplicate(
        self, embedding: np.ndarray, threshold: float = 0.6
    ) -> tuple[bool, Optional[str], Optional[float]]:
        """
        Check if embedding already exists (anti-fraud).

        Returns:
            - is_duplicate: bool
            - existing_user_id: str | None
            - similarity_score: float | None
        """
        similar = await self.find_similar(embedding, threshold=threshold, limit=1)

        if not similar:
            return False, None, None

        existing_embedding, similarity = similar[0]
        return True, existing_embedding.user_id, similarity

    async def count(self) -> int:
        """Count total embeddings."""
        result = await self.session.execute(select(func.count(Embedding.id)))
        return result.scalar_one()
