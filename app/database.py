from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:078@localhost:5432/hurry_up_hackathon"
)


engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=True)

AsyncSessionLocal = sessionmaker(
   bind=engine, 
   class_=AsyncSession, 
   expire_on_commit=False
)


async def get_db():
   async with AsyncSessionLocal() as session:
      try:
         yield session
      finally:
         await session.close()
