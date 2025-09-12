from fastapi import FastAPI
from sqlmodel import SQLModel
from contextlib import asynccontextmanager
from app.database import engine
from fastapi.middleware.cors import CORSMiddleware
from app.routers import fingerprint_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    await engine.dispose()

app = FastAPI(
    title="Hurry Up Hackathon",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Routers
app.include_router(fingerprint_service.router)

@app.get("/")
async def root():
    return {"message": "Hurry Up Hackathon API is running 🚀"}
