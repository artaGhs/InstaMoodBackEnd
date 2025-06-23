from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from routers import example
from routers import instagram_analyzer

app = FastAPI(
    title="InstaMood - Instagram Sentiment Analysis API",
    description="A FastAPI application for Instagram comment sentiment analysis using RoBERTa",
    version="1.0.0"
)

app.include_router(example.router)
app.include_router(instagram_analyzer.router)


