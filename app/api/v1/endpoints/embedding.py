import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.Repositories.embedding_repo import EmbeddingRepository

router = APIRouter()


class EmbeddingCreate(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    embedding: list[float] = Field(..., min_length=512, max_length=512)


class EmbeddingResponse(BaseModel):
    user_id: str
    created: bool
    message: str


class SimilarityCheckRequest(BaseModel):
    embedding: list[float] = Field(..., min_length=512, max_length=512)
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class SimilarityCheckResponse(BaseModel):
    is_duplicate: bool
    existing_user_id: str | None
    similarity_score: float | None


@router.post(
    "/embeddings", response_model=EmbeddingResponse, status_code=status.HTTP_201_CREATED
)
async def create_embedding(
    data: EmbeddingCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create or update embedding."""
    repo = EmbeddingRepository(db)

    # Converte para numpy
    embedding_array = np.array(data.embedding, dtype=np.float32)

    # Verifica se já existe
    existing = await repo.get_by_user_id(data.user_id)

    if existing:
        # Atualiza
        await repo.update(data.user_id, embedding_array)
        return EmbeddingResponse(
            user_id=data.user_id,
            created=False,
            message="Embedding updated successfully",
        )
    else:
        # Cria novo
        await repo.create(data.user_id, embedding_array)
        return EmbeddingResponse(
            user_id=data.user_id,
            created=True,
            message="Embedding created successfully",
        )


@router.post("/embeddings/check-duplicate", response_model=SimilarityCheckResponse)
async def check_duplicate(
    data: SimilarityCheckRequest,
    db: AsyncSession = Depends(get_db),
):
    """Check if embedding already exists (anti-fraud)."""
    repo = EmbeddingRepository(db)

    embedding_array = np.array(data.embedding, dtype=np.float32)

    is_dup, user_id, similarity = await repo.check_duplicate(
        embedding_array, threshold=data.threshold
    )

    return SimilarityCheckResponse(
        is_duplicate=is_dup,
        existing_user_id=user_id,
        similarity_score=similarity,
    )


@router.get("/embeddings/{user_id}")
async def get_embedding(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get embedding by user_id."""
    repo = EmbeddingRepository(db)
    embedding = await repo.get_by_user_id(user_id)

    if not embedding:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding not found for user_id: {user_id}",
        )

    return {
        "user_id": embedding.user_id,
        "created_at": embedding.created_at,
        "metadata": embedding.metadata,
    }


@router.delete("/embeddings/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_embedding(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete embedding."""
    repo = EmbeddingRepository(db)
    deleted = await repo.delete(user_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding not found for user_id: {user_id}",
        )
