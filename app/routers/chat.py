from uuid import UUID

from fastapi import FastAPI, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session

# Chat
router = APIRouter(prefix="/chat", tags=["Chat"])
@router.post("")
async def chat(
    query: str,
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    qdrant_client: AsyncQdrantClient = Depends(get_qdrant_async_client),
):
    try:
        answer = await rag(
            query=query,
            qdrant_client=qdrant_client,
            db=db,
            session_id=session_id,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    return answer
