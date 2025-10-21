from fastapi import FastAPI
from routes.nlp_routes import router
from config import API_PORT
import uvicorn

app = FastAPI(title="NLP Engine API")

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)