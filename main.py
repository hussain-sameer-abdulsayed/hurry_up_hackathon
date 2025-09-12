from fastapi import Depends, FastAPI, UploadFile
from sqlmodel import SQLModel
import uvicorn
from app.database import engine, get_db
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
import os
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
   async with engine.begin() as conn:
      await conn.run_sync(SQLModel.metadata.create_all)
   yield
   await engine.dispose()


app = FastAPI(
   title= os.getenv("APP_NAME", "Hurry Up Hackathon"),
   description="Best Team",
   version="1",
   lifespan=lifespan
)


app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)



@app.get("/")
async def root():
   return {
      "Hurry Up Hackathon": "Best Team Ever! ok?",
      "docs": "/docs",
      "redoc": "/redoc",
      "version": "1.0.0",
      "environment": os.getenv("APP_ENV", "development")
   }

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    return {"status": "healthy", "database": "connected"}


@app.post("/")
async def upload_file(
   file: UploadFile,
):
   return file



if __name__ == "__main__":
   uvicorn.run(
      "main:app",
      host="0.0.0.0",
      port=os.getenv("PORT", 8000),
      reload=os.getenv("APP_ENV") != "production"
   )