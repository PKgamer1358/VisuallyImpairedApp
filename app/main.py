from fastapi import FastAPI
from app.routes import users, object_detection
from app.database import Base, engine

# Create database tables (only for the user example — safe to keep)
Base.metadata.create_all(bind=engine)

# Initialize FastAPI
app = FastAPI(title="Object Detection API")

# Include routes
app.include_router(users.router)
app.include_router(object_detection.router)

# Optional: Root endpoint for sanity check
@app.get("/")
def root():
    return {"message": "Object Detection API is running 🚀"}