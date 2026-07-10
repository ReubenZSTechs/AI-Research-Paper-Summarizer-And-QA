import asyncio
import logging
import threading

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool



from backend.api.schema.LLMResponse import ChatQuery, ChatResponse
from backend.api.schema.RAGResponse import RAGQuery, RAGResponse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=16)
processing_lock = Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Connecting to vLLM servers")


    logger.info(f"Connected to vLLM servers")
    yield
    logger.info(f"Server shutting down")
    executor.shutdown(wait=False)
    


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True)


@app.post("/llm/chat")
async def chat(query: ChatQuery):
    acquired = processing_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=429, detail=f"Server is busy. Please wait...")
    
    try:
        user_query = query.query


    except asyncio.TimeoutError:
        logger.error(f"[LLM] Request timed out")
        raise HTTPException(status_code=504, detail=f"Request timed out due to LLM")
    
    except Exception as e:
        logger.error(f"[LLM] Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error occured: {e}")
    
    finally:
        if acquired:
            processing_lock.release()