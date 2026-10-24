from book_rag_backend.api.routes.documents import router as documents_router
from book_rag_backend.api.routes.search import router as search_router

__all__ = ["search_router", "documents_router"]
