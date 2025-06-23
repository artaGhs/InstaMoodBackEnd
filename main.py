from fastapi import FastAPI
from routers import instagram_analyzer

app = FastAPI(
    title="InstaMood - Instagram Sentiment Analysis API",
    description="A FastAPI application for Instagram comment sentiment analysis using RoBERTa",
    version="1.0.0"
)

app.include_router(instagram_analyzer.router)


