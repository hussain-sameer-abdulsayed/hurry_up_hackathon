from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import desc, select

from app.models.finger_print import FingerPrint
from app.schemas.finger_print import FingerPrintCreate


class FingerPrintRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self, data: FingerPrintCreate) -> FingerPrint:
        fingerprint = FingerPrint(**data.model_dump())
        self.db.add(fingerprint)
        await self.db.commit()
        await self.db.refresh(fingerprint)
        return fingerprint
    
    async def get_all(self) -> List[FingerPrint]:
        statement = select(FingerPrint).order_by(desc(FingerPrint.created_at))
        result = await self.db.execute(statement)
        return list(result.scalars().all())
    

    async def delete_all(self) -> int:
      """Delete all fingerprints and return count"""
      statement = select(FingerPrint)
      result = await self.db.execute(statement)
      fingerprints = result.scalars().all()
      count = len(fingerprints)
      
      for fp in fingerprints:
         await self.db.delete(fp)
      
      await self.db.commit()
      return count