from fastapi import APIRouter, status

router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "aletheia-api",
    }


@router.get("/health/db", status_code=status.HTTP_200_OK)
async def health_check_db():
    """Database health check."""
    # TODO: Testar conexão com PostgreSQL
    print("Mocado")
    return {
        "status": "healthy",
        "database": "postgresql",
        "connected": True,
    }
