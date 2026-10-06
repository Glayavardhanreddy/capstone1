"""SQLAlchemy database models for Users and AutoML Experiments."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """User account model for authentication and experiment ownership."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    experiments = relationship("Experiment", back_populates="owner", cascade="all, delete-orphan")


class Experiment(Base):
    """AutoML optimization experiment record linking dataset, models, and results."""
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    title = Column(String(120), nullable=False)
    dataset_name = Column(String(120), nullable=False)
    task_type = Column(String(50), nullable=False)
    target_column = Column(String(80), nullable=False)
    n_samples = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    primary_metric = Column(String(50), nullable=False)
    best_model_name = Column(String(120), nullable=False)
    best_config_name = Column(String(120), nullable=False)
    best_score = Column(Float, nullable=False)
    leaderboard_json = Column(Text, nullable=False)      # JSON string
    meta_features_json = Column(Text, nullable=False)    # JSON string
    diagnostics_json = Column(Text, nullable=True)       # JSON string
    model_file_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="experiments")
