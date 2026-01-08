from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "aletheia-api",
    }


@router.get("/health/db", status_code=status.HTTP_200_OK)
async def health_check_db(db: AsyncSession = Depends(get_db)):
    """Database health check."""
    try:
        await db.execute(text("SELECT 1"))

        result = await db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        has_pgvector = result.scalar_one_or_none() is not None

        return {
            "status": "healthy",
            "database": "postgresql",
            "connected": True,
            "pgvector": has_pgvector,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "postgresql",
            "connected": False,
            "error": str(e),
        }
