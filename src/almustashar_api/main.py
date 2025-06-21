import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

# Assuming config.py is in the same directory or accessible via python path
from .config import API_TITLE, API_VERSION, CORS_ALLOWED_ORIGINS, LOG_LEVEL

# Import the Almustashar agent graph
# Adjust the import path based on your project structure
# Assuming almustashar_api is a sub-package of src, and retrieval_graph is also a sub-package of src
from ..retrieval_graph.graph import graph as almustashar_compiled_agent # Import the compiled graph instance
from ..retrieval_graph.state import AgentState # For type hinting if needed

# --- Logging Setup ---
# Configure logging (ensure this doesn't conflict with existing logging setup)
# Basic config if no handlers are present
if not logging.root.handlers:
    logging.basicConfig(level=LOG_LEVEL)
    print(f"INFO:     Logging configured with level: {LOG_LEVEL}")

# Quieten noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)
logging.getLogger('supabase').setLevel(logging.WARNING) # Add supabase if it becomes noisy
logging.getLogger('langgraph').setLevel(logging.WARNING) # Changed from INFO to WARNING
logging.getLogger('custom_nodes').setLevel(logging.WARNING)
logging.getLogger('retrieval_graph').setLevel(logging.WARNING)
logging.getLogger('shared').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Lifespan Management for Agent Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("INFO:     Application startup: Initializing Almustashar agent...")
    try:
        app.state.almustashar_agent = almustashar_compiled_agent # Assign the imported instance
        logger.info("INFO:     Application startup: Almustashar agent initialized successfully.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Almustashar agent during startup: {e}")
        logger.error(traceback.format_exc())
        # Optionally, re-raise or handle to prevent app from starting if agent is critical
        # For now, we'll let it start but log the critical failure.
        app.state.almustashar_agent = None # Ensure it's None if failed
    yield
    logger.info("INFO:     Application shutdown.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range"] # If your client needs it for pagination, etc.
)
logger.info(f"INFO:     CORS Middleware configured for origins: {CORS_ALLOWED_ORIGINS}")

# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    logger.error(traceback.format_exc())

    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_type": type(exc).__name__},
        )
    
    # For any other unhandled exceptions, return a generic 500
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred.", "error_type": type(exc).__name__},
    )

# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def read_root():
    """
    Root endpoint for basic health check.
    Indicates if the API is running and if the agent was initialized.
    """
    agent_status = "initialized" if hasattr(app.state, 'almustashar_agent') and app.state.almustashar_agent else "initialization_failed"
    return {
        "message": f"{API_TITLE} is running.",
        "version": API_VERSION,
        "almustashar_agent_status": agent_status
    }

# --- Routers (to be added later) ---
from .routers import rag_almustashar_router
app.include_router(rag_almustashar_router.router)

# --- Main guard for running with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # Host and port can be configured via environment variables or a config file
    # For development, 0.0.0.0 makes it accessible on the network
    # The config.py should ideally handle these, but for direct run:
    server_host = os.getenv("API_HOST", "0.0.0.0")
    server_port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting Uvicorn server on {server_host}:{server_port}")
    uvicorn.run(app, host=server_host, port=server_port)
