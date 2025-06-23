from fastapi import HTTPException, APIRouter, UploadFile, File, Form
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np
from typing import List, Optional, Union
import re
import os
import tempfile
import subprocess
import whisper
import yt_dlp

import librosa
import soundfile as sf
from pathlib import Path
import asyncio

router = APIRouter(
    prefix="/instagram",
    tags=["instagram-video-sentiment"]
)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

class InstagramVideoRequest(BaseModel):
    video_url: str = Field(description="Instagram video/reel URL")
    transcribe_language: Optional[str] = Field(default=None, description="Language for transcription (auto-detect if None)")

class VideoSentimentResponse(BaseModel):
    video_url: Optional[str]
    transcription: str
    transcription_confidence: float
    sentiment_scores: dict
    predicted_sentiment: str
    text_segments: List[dict] = Field(description="Timestamped transcription segments")

class UploadVideoRequest(BaseModel):
    transcribe_language: Optional[str] = Field(default=None, description="Language for transcription")

def extract_instagram_video_url(instagram_url: str) -> str:
    """Extract the actual video URL from Instagram post"""
    try:
        # Configure yt-dlp for Instagram
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'noplaylist': True,
            'extract_flat': False,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(instagram_url, download=False)
            if 'url' in info:
                return info['url']
            elif 'entries' in info and len(info['entries']) > 0:
                return info['entries'][0]['url']
            else:
                raise ValueError("Could not extract video URL")
                
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract video URL: {str(e)}")

def download_instagram_video(instagram_url: str, output_path: str) -> str:
    """Download Instagram video using yt-dlp"""
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([instagram_url])
            
        return output_path
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

def extract_audio_from_video(video_path: str, audio_path: str) -> str:
    """Extract audio from video file using ffmpeg."""
    try:
        subprocess.run([
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', audio_path, '-y'
        ], check=True, capture_output=True, text=True)
        return audio_path
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, 
            detail="ffmpeg not found. Please ensure ffmpeg is installed and in your PATH."
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract audio: {e.stderr}")

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> dict:
    """Transcribe audio using Whisper"""
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )
        
        return {
            'text': result['text'].strip(),
            'language': result['language'],
            'segments': result.get('segments', []),
            'confidence': np.mean([seg.get('no_speech_prob', 0.5) for seg in result.get('segments', [{'no_speech_prob': 0.5}])])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")

def analyze_text_sentiment(text: str) -> dict:
    """Analyze sentiment of transcribed text"""
    try:
        if not text or not text.strip():
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
            
        # Clean the text
        cleaned_text = text.strip()
        
        # Split long text into chunks if needed (RoBERTa has token limits)
        max_length = 500
        if len(cleaned_text) > max_length:
            # Split into sentences and analyze each
            sentences = re.split(r'[.!?]+', cleaned_text)
            sentence_scores = []
            
            for sentence in sentences:
                if sentence.strip():
                    encoded_text = tokenizer(sentence.strip(), return_tensors='pt', truncation=True, max_length=512)
                    with torch.no_grad():
                        output = model(**encoded_text)
                    scores = output[0][0].detach().numpy()
                    scores = softmax(scores)
                    sentence_scores.append({
                        'negative': float(scores[0]),
                        'neutral': float(scores[1]),
                        'positive': float(scores[2])
                    })
            
            # Average the scores
            if sentence_scores:
                avg_scores = {
                    'negative': float(np.mean([s['negative'] for s in sentence_scores])),
                    'neutral': float(np.mean([s['neutral'] for s in sentence_scores])),
                    'positive': float(np.mean([s['positive'] for s in sentence_scores]))
                }
                return avg_scores
        
        # Analyze the full text if it's short enough
        encoded_text = tokenizer(cleaned_text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            output = model(**encoded_text)
        
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        return {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2])
        }
        
    except Exception as e:
        print(f"Error analyzing sentiment: {str(e)}")
        return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}

@router.get("/")
async def root():
    return {
        "message": "Instagram Video Sentiment Analysis API is running!",
        "description": "Analyze sentiment of Instagram videos/reels by transcribing their audio content",
        "endpoints": {
            "/analyze_video_url": "Analyze Instagram video from URL",
            "/analyze_uploaded_video": "Analyze uploaded video file",
            "/demo": "Try with sample text",
            "/health": "Check API health"
        },
        "supported_formats": ["Instagram Reels", "Instagram Videos", "IGTV", "MP4 uploads"],
        "note": "This service downloads video, extracts audio, transcribes it, and analyzes sentiment"
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "sentiment_model": MODEL,
        "transcription_model": "whisper-base",
        "supported_languages": "auto-detect + 99 languages"
    }

@router.post("/analyze_video_url", response_model=VideoSentimentResponse)
async def analyze_instagram_video(request: InstagramVideoRequest):
    """Analyze sentiment of Instagram video by transcribing its audio"""
    temp_dir = None
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "video.mp4")
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        print(f"Processing Instagram video: {request.video_url}")
        
        # Download the video
        print("Downloading video...")
        download_instagram_video(request.video_url, video_path)
        
        # Extract audio
        print("Extracting audio...")
        extract_audio_from_video(video_path, audio_path)
        
        # Transcribe audio
        print("Transcribing audio...")
        transcription_result = transcribe_audio(audio_path, request.transcribe_language)
        
        if not transcription_result['text']:
            raise HTTPException(status_code=400, detail="No speech detected in the video")
        
        print(f"Transcription: {transcription_result['text'][:100]}...")
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_scores = analyze_text_sentiment(transcription_result['text'])
        
        # Determine predicted sentiment
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[np.argmax([sentiment_scores['negative'], sentiment_scores['neutral'], sentiment_scores['positive']])]
        
        # Format segments for response
        text_segments = []
        for segment in transcription_result.get('segments', []):
            text_segments.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', ''),
                'confidence': 1.0 - segment.get('no_speech_prob', 0.5)
            })
        
        return VideoSentimentResponse(
            video_url=request.video_url,
            transcription=transcription_result['text'],
            transcription_confidence=1.0 - transcription_result['confidence'],
            sentiment_scores=sentiment_scores,
            predicted_sentiment=predicted_sentiment,
            text_segments=text_segments
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

@router.post("/analyze_uploaded_video", response_model=VideoSentimentResponse)
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    transcribe_language: Optional[str] = Form(None)
):
    """Analyze sentiment of uploaded video file"""
    temp_dir = None
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Please upload a video file")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"uploaded_video_{file.filename}")
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        print(f"Processing uploaded video: {file.filename}")
        
        # Save uploaded file
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract audio
        print("Extracting audio...")
        extract_audio_from_video(video_path, audio_path)
        
        # Transcribe audio
        print("Transcribing audio...")
        transcription_result = transcribe_audio(audio_path, transcribe_language)
        
        if not transcription_result['text']:
            raise HTTPException(status_code=400, detail="No speech detected in the video")
        
        print(f"Transcription: {transcription_result['text'][:100]}...")
        
        # Analyze sentiment
        print("Analyzing sentiment...")
        sentiment_scores = analyze_text_sentiment(transcription_result['text'])
        
        # Determine predicted sentiment
        sentiment_labels = ['negative', 'neutral', 'positive']
        predicted_sentiment = sentiment_labels[np.argmax([sentiment_scores['negative'], sentiment_scores['neutral'], sentiment_scores['positive']])]
        
        # Format segments for response
        text_segments = []
        for segment in transcription_result.get('segments', []):
            text_segments.append({
                'start': segment.get('start', 0),
                'end': segment.get('end', 0),
                'text': segment.get('text', ''),
                'confidence': 1.0 - segment.get('no_speech_prob', 0.5)
            })
        
        return VideoSentimentResponse(
            video_url=f"uploaded:{file.filename}",
            transcription=transcription_result['text'],
            transcription_confidence=1.0 - transcription_result['confidence'],
            sentiment_scores=sentiment_scores,
            predicted_sentiment=predicted_sentiment,
            text_segments=text_segments
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded video: {str(e)}")
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

@router.get("/demo")
async def demo_analysis():
    """Demo endpoint with sample transcribed text analysis"""
    sample_text = """
    Hey everyone! I'm so excited to share this new recipe with you. 
    It's absolutely delicious and I think you're going to love it. 
    The flavors are incredible and it's so easy to make. 
    Let me know in the comments what you think!
    """
    
    sentiment_scores = analyze_text_sentiment(sample_text)
    sentiment_labels = ['negative', 'neutral', 'positive']
    predicted_sentiment = sentiment_labels[np.argmax([sentiment_scores['negative'], sentiment_scores['neutral'], sentiment_scores['positive']])]
    
    return VideoSentimentResponse(
        video_url="demo://sample-video",
        transcription=sample_text.strip(),
        transcription_confidence=0.95,
        sentiment_scores=sentiment_scores,
        predicted_sentiment=predicted_sentiment,
        text_segments=[
            {"start": 0.0, "end": 5.0, "text": "Hey everyone! I'm so excited to share this new recipe with you.", "confidence": 0.95},
            {"start": 5.0, "end": 10.0, "text": "It's absolutely delicious and I think you're going to love it.", "confidence": 0.93},
            {"start": 10.0, "end": 15.0, "text": "The flavors are incredible and it's so easy to make.", "confidence": 0.97},
            {"start": 15.0, "end": 18.0, "text": "Let me know in the comments what you think!", "confidence": 0.91}
        ]
    )

@router.get("/instructions")
async def get_instructions():
    """Get detailed instructions for video sentiment analysis"""
    return {
        "title": "Instagram Video Sentiment Analysis Instructions",
        "description": "Analyze sentiment of Instagram videos by transcribing their audio content",
        "methods": [
            {
                "method": "Instagram URL Analysis",
                "endpoint": "/analyze_video_url",
                "description": "Provide an Instagram video/reel URL for analysis",
                "example": "POST /instagram/analyze_video_url",
                "supported_urls": [
                    "https://www.instagram.com/reel/...",
                    "https://www.instagram.com/p/... (videos)",
                    "https://www.instagram.com/tv/..."
                ]
            },
            {
                "method": "Video File Upload",
                "endpoint": "/analyze_uploaded_video",
                "description": "Upload a video file directly",
                "example": "POST /instagram/analyze_uploaded_video",
                "supported_formats": ["MP4", "MOV", "AVI", "MKV"]
            }
        ],
        "process": [
            "1. Video is downloaded/received",
            "2. Audio is extracted from video",
            "3. Audio is transcribed using Whisper AI",
            "4. Transcribed text is analyzed for sentiment",
            "5. Results include both transcription and sentiment analysis"
        ],
        "features": [
            "Multi-language transcription support",
            "Timestamped transcription segments",
            "Confidence scores for transcription",
            "Detailed sentiment analysis",
            "Support for long videos (chunked analysis)"
        ]
    }

