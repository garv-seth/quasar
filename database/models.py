import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os

# Get database URL from environment
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)  # 'pending', 'completed', 'failed'
    result = Column(JSON)
    execution_time = Column(Float)  # in seconds
    quantum_metrics = relationship("QuantumMetrics", back_populates="task")

class QuantumMetrics(Base):
    __tablename__ = "quantum_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("tasks.id"))
    quantum_advantage = Column(Float)  # percentage improvement
    memory_efficiency = Column(Float)  # percentage improvement
    circuit_depth = Column(Integer)
    qubit_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    task = relationship("Task", back_populates="quantum_metrics")

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
