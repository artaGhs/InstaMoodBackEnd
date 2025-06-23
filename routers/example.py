from fastapi import HTTPException, APIRouter
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch


router = APIRouter(
    prefix="/api",
    tags=["sentiment-analysis"]
)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

class SentimentRequest(BaseModel):
    text: str = Field(max_length=250)

class SentimentResponse(BaseModel):
    text: str 
    sentiment_scores: dict
    predicted_sentiment: str

@router.get("/")
async def root():
    return {"message": "RoBERTa Sentiment Analysis API is running!"}

@router.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL}

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        # Tokenize the input text
        encoded_text = tokenizer(request.text, return_tensors='pt')
        
        # Get model predictions
        with torch.no_grad():
            output = model(**encoded_text)
        
        # Process the scores
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        # Create scores dictionary
        scores_dict = {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2])
        }
        
        # Determine the predicted sentiment
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[scores.argmax()]
        
        return SentimentResponse(
            text=request.text,
            sentiment_scores=scores_dict,
            predicted_sentiment=predicted_sentiment
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")
    

@router.post("/analyze_batch")
async def analyze_sentiment_batch(texts: list[str]):
    try:
        results = []
        for text in texts:
            encoded_text = tokenizer(text, return_tensors='pt')
            
            with torch.no_grad():
                output = model(**encoded_text)
            
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            
            scores_dict = {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }
            
            sentiment_labels = ['negative', 'neutral', 'positive']
            predicted_sentiment = sentiment_labels[scores.argmax()]
            
            results.append({
                'text': text,
                'sentiment_scores': scores_dict,
                'predicted_sentiment': predicted_sentiment
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")