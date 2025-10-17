# backend.py
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import sqlite3
from datetime import datetime, timedelta, timezone
import uvicorn
import json
import asyncio
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import logging
import websockets
from dotenv import load_dotenv
from openai import OpenAI
import httpx
import re
import os
from contextlib import asynccontextmanager
from enum import Enum

load_dotenv()

# Configuration and ML model load
try:
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(DATA_DIR, "emotion_predictor.pkl")  # ✅ Fixed typo
    ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.pkl")    # ✅ Fixed = to ()
    UNIFIED_FILE_TPL = os.path.join(DATA_DIR, "user_{uid}_unified.pkl")
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI client initialized")
    else:
        client = None
        logger.warning("⚠️ OPENAI_API_KEY not found. AI features will be limited.")
except Exception as e:
    client = None
    logger.error(f"Failed to initialize OpenAI: {e}")

DATABASE = "trading_app.db"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global model objects + locks
model: Optional[RandomForestClassifier] = None
le: Optional[LabelEncoder] = None
model_lock = asyncio.Lock()
save_lock = asyncio.Lock()




# Initialize ML model and label encoder
model = RandomForestClassifier(n_estimators=100, random_state=42)
le = LabelEncoder()
le.fit(["neutral", "fomo", "revenge", "greed", "fear"])

# for deploytation enhancement from the above
async def load_or_init_model():
    global model, le
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        logger.info("Loaded existing model & encoder")
    else:
        logger.info("No saved model found. Training fallback model...")
        model, le = model, le
        logger.info("Fallback model trained & saved")


# In-memory storage for users, moods, journal
users = {}
moods = {}
journal = {}

# Pydantic models
class PerformanceMetric(BaseModel):
    id: str
    metric: str
    value: str
    change: str
    isPositive: bool
    icon: str

class PerformanceDistribution(BaseModel):
    profit: int
    loss: int
    breakEven: int

class MoodEntry(BaseModel):
    id: str
    day: str
    mood: str
    emoji: str
    note: str

class EmotionalTriggerMetric(BaseModel):
    icon: str
    label: str
    value: int
    color: str
    description: str

class EmotionalTriggerData(BaseModel):
    metrics: List[EmotionalTriggerMetric]
    radarData: Dict[str, float]
    emotionalScore: float

class AIInsight(BaseModel):
    id: str
    type: str
    title: str
    description: str
    icon: str
    color: str
    bgColor: str

class Recommendation(BaseModel):
    id: str
    text: str

class FeedbackItem(BaseModel):
    id: int
    type: str
    icon: str
    title: str
    message: str
    priority: str
    bgColor: str
    iconColor: str

class RecentActivity(BaseModel):
    id: str
    type: str
    title: str
    description: str
    timestamp: str
    icon: str
    color: str
    value: str

# merge Trade and TradeRequest
class Trade(BaseModel):
    userId: str
    inToken: str
    outToken: str
    amountIn: float
    amountOut: float
    volumeUsd: float

class TradeRequest(BaseModel):
    userId: str
    inToken: str
    outToken: str
    amountIn: float
    amountOut: float
    volumeUsd: float
    orderType: str
    leverage: int
    price: float | None = None
    mode: str = "live"
    slippage_factor: Optional[float] = Field(default=1.0, gt=0)
    latency_ms: Optional[int] = Field(default=0, ge=0)
    timestamp: Optional[datetime] = None

    @field_validator("timestamp")
    @classmethod
    def _default_ts(cls, v):
        return v or datetime.now(timezone.utc)


class UserTrade(BaseModel):
    id: str
    user_id: str
    wallet: str
    in_token: str
    out_token: str
    amount_in: float
    amount_out: float
    volume_usd: float
    timestamp: str
    emotion: str
    trigger_details: str | None
    entry_price: float
    exit_price: float
    pnl: float

class ChartDataPoint(BaseModel):
    id: str
    timestamp: str
    value: float

class EmotionalGaugeData(BaseModel):
    emotionalScore: float
    maxScore: float
    breakdown: Dict[str, Dict[str, float | str]]

class EmotionalScoreGaugeData(BaseModel):
    score: float
    breakdown: Dict[str, Dict[str, float | str]]

class EmotionRequest(BaseModel):
    userId: str
    currentEmotion: str
    confidence: int
    fear: int
    excitement: int
    stress: int

class Mood(BaseModel):
    user_id: str
    mood: str
    timestamp: str

class FeedbackRequest(BaseModel):
    userId: str
    symbol: str
    mode: str

class JournalEntry(BaseModel):
    user_id: str
    entry: str
    timestamp: str

class ConnectWalletRequest(BaseModel):
    walletAddress: str
    walletType: str  # "metamask", "phantom", "internet_identity"

# ArcheType models
class Archetype(str, Enum):
    FOMO_APE = "fomo_ape"
    GREEDY_BULL = "greedy_bull"
    FEARFUL_WHALE = "fearful_whale"
    RATIONAL_TRADER = "rational_trader"
    REVENGE_TRADER = "revenge_trader"
    PATIENT_HODLER = "patient_hodler"

class ArchetypeResponse(BaseModel):
    user_id: str
    archetype: Archetype
    confidence: float
    traits: Dict[str, float]
    description: str
    recommendations: List[str]

class TradingPatterns(BaseModel):
    avg_trade_frequency_min: float
    win_rate: float
    avg_hold_time_hours: float
    risk_reward_ratio: float
    emotional_volatility: float
    consecutive_losses: int
    fomo_score: float
    greed_score: float
    fear_score: float

# Market Trend Models
class TechnicalIndicators(BaseModel):
    symbol: str
    rsi: float
    rsi_signal: str  # overbought, oversold, neutral
    breakout_detected: bool
    breakout_strength: float
    volume_spike: bool
    trend_direction: str  # bullish, bearish, neutral
    support_level: Optional[float]
    resistance_level: Optional[float]
    confidence: float

class MarketAnalysisRequest(BaseModel):
    symbols: List[str]
    exchange: str = "binance"  # binance, solana, duni
    timeframe: str = "1h"  # 1m, 5m, 15m, 1h, 4h, 1d

class MarketAnalysisResponse(BaseModel):
    user_id: str
    timestamp: str
    analysis: Dict[str, TechnicalIndicators]


#  Functions
def validate_wallet_address(wallet_type: str, address: str) -> bool:
    if wallet_type == "internet_identity":
        # ICP principal: 53 characters, base32
        return bool(re.match(r'^[a-z0-9\-]{53}$', address))
    else:
        # Ethereum/Solana: 0x-prefixed hex (40 chars) or Solana base58 (44 chars)
        return bool(re.match(r'^0x[a-fA-F0-9]{40}$|^[1-9A-HJ-NP-Za-km-z]{32,44}$', address))

# Feature engineering function
def engineer_features(df):
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / 60
    df['slippage_factor'] = 0.99
    df['price_in'] = (df['volume_usd'] / df['amount_in']) * df['slippage_factor']
    df['price_out'] = (df['volume_usd'] / df['amount_out'].replace(0, 42490798.02)) * df['slippage_factor']
    df['market_price'] = df['price_out'].rolling(window=60).mean()
    df['price_change_pct'] = ((df['price_out'] - df['market_price']) / df['market_price']) * 100
    df['account_equity'] = 1000
    df['leverage'] = (df['amount_in'] * df['price_in']) / df['account_equity']
    df['position_change'] = df['leverage'].pct_change()
    df['trade_pair'] = df['in_token'] + '_' + df['out_token']
    df['entry_price'] = df.groupby('trade_pair')['price_in'].shift(1)
    df['exit_price'] = df['price_out']
    df['pnl'] = (df['exit_price'] - df['entry_price']) * df['amount_out'] * 0.997
    df['is_win'] = df['pnl'] > 0
    df['is_loss'] = df['pnl'] < 0
    df['win_streak'] = (df['pnl'] > 0).groupby((df['pnl'] <= 0).cumsum()).cumcount() + 1
    df['loss_streak'] = (df['pnl'] < 0).groupby((df['pnl'] >= 0).cumsum()).cumcount() + 1
    df['consecutive_losses'] = (df['pnl'] < 0).rolling(window=3).sum()
    df['consecutive_wins'] = (df['pnl'] > 0).rolling(window=3).sum()
    return df.fillna(0)  # Handle NaN values

# Emotion detection
def detect_emotion(df):
    df['emotion'] = 'neutral'
    df['trigger_details'] = None
    for i in range(1, len(df)):
        prev_row, curr_row = df.iloc[i-1], df.iloc[i]
        if (curr_row['price_change_pct'] > 20 and curr_row['time_diff'] < 30 and curr_row['win_streak'] > 1):
            df.loc[i, 'emotion'] = 'fomo'
            df.loc[i, 'trigger_details'] = f"Price spike: {curr_row['price_change_pct']:.2f}%, win streak: {curr_row['win_streak']}"
        elif (curr_row['time_diff'] < 2 and prev_row['consecutive_losses'] >= 2):
            df.loc[i, 'emotion'] = 'revenge'
            df.loc[i, 'trigger_details'] = f"After {prev_row['consecutive_losses']} losses, latency: {curr_row['time_diff']}"
        elif (curr_row['position_change'] > 0.5 and prev_row['consecutive_wins'] >= 2):
            df.loc[i, 'emotion'] = 'greed'
            df.loc[i, 'trigger_details'] = f"After {prev_row['consecutive_wins']} wins, pos change: {curr_row['position_change']:.2f}"
        elif (curr_row['time_diff'] < 1 and curr_row['pnl'] > 0 and abs(curr_row['pnl']) < 10):
            df.loc[i, 'emotion'] = 'fear'
            df.loc[i, 'trigger_details'] = f"Early close: {curr_row['pnl']:.2f}, latency: {curr_row['time_diff']}"
    return df

# ---------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------
def call_openai_warning(df: pd.DataFrame, predicted_emotion: str) -> Dict[str, Any]:
    """Ask OpenAI to turn features into a human-friendly warning."""
    if client is None or not OPENAI_API_KEY:  # ✅ Check client, not openai module
        return {
            "insight": f"Heuristic-only: model suggests {predicted_emotion}.",
            "warning": "Trading psychology signal generated without OpenAI.",
            "recommendation": "Reduce leverage, avoid rapid entries, set stops.",
            "advice": "Take a short break before placing the next trade.",
        }

    try:
        prompt = (
            "You are a trading-psychology assistant. Given the following last records "
            "(spot and/or futures), summarize a concise warning and recommendation in JSON "
            "with keys: insight, warning, recommendation, advice.\n\n"
            f"Predicted emotion: {predicted_emotion}\n"
            f"Records (tail 10 rows):\n{df.tail(10).to_json(orient='records')}"
        )
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content if resp and resp.choices else "{}"
        try:
            parsed = json.loads(content)
            for k in ["insight", "warning", "recommendation", "advice"]:
                parsed.setdefault(k, "")
            return parsed
        except Exception:
            return {
                "insight": f"{predicted_emotion} detected. Keep risk in check.",
                "warning": "LLM returned non-JSON; using safe defaults.",
                "recommendation": "Tighten risk, avoid overtrading.",
                "advice": "Journal your thoughts and wait 15 minutes.",
            }
    except Exception as e:
        logger.warning(f"OpenAI call failed: {e}")
        return {
            "insight": f"{predicted_emotion} behavior inferred.",
            "warning": "OpenAI unavailable; heuristic advice only.",
            "recommendation": "Lower size and set clear invalidation.",
            "advice": "Step away briefly, review your plan.",
        }

# Prepare ML data from last 3 trades
def prepare_ml_data(df):
    features = ['time_diff', 'price_change_pct', 'position_change', 'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak']
    ml_data = []

    for col in features:
        if col not in df.columns:
            df[col] = 0

    for i in range(2, len(df)):
        prev3 = df.iloc[i-2:i+1].copy() # Or df.iloc[i - 3 : i] would give 4 rows
        curr = df.iloc[i]
        if len(prev3) == 3:
            ml_data.append({
                'time_diff': prev3['time_diff'].mean(),
                'price_change_pct': prev3['price_change_pct'].mean(),
                'position_change': prev3['position_change'].mean(),
                'consecutive_wins': prev3['consecutive_wins'].iloc[-1],
                'consecutive_losses': prev3['consecutive_losses'].iloc[-1],
                'win_streak': prev3['win_streak'].iloc[-1],
                'loss_streak': prev3['loss_streak'].iloc[-1],
                'emotion': curr['emotion']
            })
    return pd.DataFrame(ml_data) if ml_data else pd.DataFrame()

# Warning and recommendation generation
async def get_emotion_warning(wallet_df, predicted_emotion):
    recent_trades = wallet_df.tail(3).to_dict(orient='records')
    prompt = f"""
    User with wallet {wallet_df['wallet'].iloc[0]} has shown a predicted emotion of {predicted_emotion} based on their last 3 trades. 
    Trade data: {json.dumps(recent_trades, default=str)}
    Provide insight, warning, recommendation, and advice in JSON format.
    """
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                timeout=30
            )
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return {
            "insight": "Error fetching insight",
            "warning": "Unable to generate warning",
            "recommendation": "None",
            "advice": "Contact support"
        }

# Helper function for emotion prediction
def predict_next_emotion(df, model, le):
    if len(df) < 4:
        return "neutral", "Not enough trades"
    last3 = df.iloc[-3:].copy()
    features = {
        'time_diff': last3['time_diff'].mean(),
        'price_change_pct': last3['price_change_pct'].mean(),
        'position_change': last3['position_change'].mean(),
        'consecutive_wins': last3['consecutive_wins'].iloc[-1],
        'consecutive_losses': last3['consecutive_losses'].iloc[-1],
        'win_streak': last3['win_streak'].iloc[-1],
        'loss_streak': last3['loss_streak'].iloc[-1]
    }
    X_pred = pd.DataFrame([features])
    pred_encoded = model.predict(X_pred)[0]
    emotion = le.inverse_transform([pred_encoded])[0]
    return emotion, "Prediction based on last 3 trades"

# Database setup
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id TEXT PRIMARY KEY,
            metric TEXT,
            value TEXT,
            change TEXT,
            isPositive INTEGER,
            icon TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_distribution (
            id TEXT PRIMARY KEY,
            profit INTEGER,
            loss INTEGER,
            breakEven INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_entries (
            id TEXT PRIMARY KEY,
            day TEXT,
            mood TEXT,
            emoji TEXT,
            note TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotional_triggers (
            id TEXT PRIMARY KEY,
            icon TEXT,
            label TEXT,
            value INTEGER,
            color TEXT,
            description TEXT,
            greed REAL,
            confidence REAL,
            fear REAL,
            revenge REAL,
            fomo REAL,
            emotionalScore REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_insights (
            id TEXT PRIMARY KEY,
            type TEXT,
            title TEXT,
            description TEXT,
            icon TEXT,
            color TEXT,
            bgColor TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            text TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY,
            type TEXT,
            icon TEXT,
            title TEXT,
            message TEXT,
            priority TEXT,
            bgColor TEXT,
            iconColor TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recent_activities (
            id TEXT PRIMARY KEY,
            type TEXT,
            title TEXT,
            description TEXT,
            timestamp TEXT,
            icon TEXT,
            color TEXT,
            value TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            wallet TEXT,
            in_token TEXT,
            out_token TEXT,
            amount_in REAL,
            amount_out REAL,
            volume_usd REAL,
            timestamp TEXT,
            emotion TEXT,
            trigger_details TEXT,
            entry_price REAL,
            exit_price REAL,
            pnl REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chart_data (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            value REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotional_gauge (
            id TEXT PRIMARY KEY,
            emotionalScore REAL,
            maxScore REAL,
            disciplineScore REAL,
            disciplineColor TEXT,
            patienceScore REAL,
            patienceColor TEXT,
            riskControlScore REAL,
            riskControlColor TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotional_score_gauge (
            id TEXT PRIMARY KEY,
            score REAL,
            fearPercentage REAL,
            fearColor TEXT,
            greedPercentage REAL,
            greedColor TEXT,
            confidencePercentage REAL,
            confidenceColor TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            strategy_data TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            balance REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            in_token TEXT,
            out_token TEXT,
            amount_in REAL,
            amount_out REAL,
            volume_usd REAL,
            entry_price REAL,
            fee REAL,
            emotion TEXT,
            timestamp TEXT,
            pnl REAL DEFAULT 0
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            current_emotion TEXT,
            confidence INTEGER,
            fear INTEGER,
            excitement INTEGER,
            stress INTEGER,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            symbol TEXT,
            type TEXT,
            message TEXT,
            accuracy INTEGER,
            timestamp TEXT,
            mode TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize database
init_db()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load / init model
    await load_or_init_model()

    # Background retraining task (configurable interval)
    interval_hours = int(os.getenv("MODEL_RETRAIN_INTERVAL_HOURS", "24"))
    interval_seconds = max(1, interval_hours * 3600)

    async def background_retrainer():
        nonlocal interval_seconds
        while True:
            await asyncio.sleep(interval_seconds)
            logger.info(f"\U0001F501 Scheduled retraining (every {interval_hours}h)...")
            try:
                # Load all user unified datasets and retrain
                dfs = []
                for fname in os.listdir(DATA_DIR):
                    if fname.startswith("user_") and fname.endswith("_unified.pkl"):
                        try:
                            dfs.append(joblib.load(os.path.join(DATA_DIR, fname)))
                        except Exception as e:
                            logger.warning(f"Couldn't load {fname}: {e}")
                if not dfs:
                    logger.info("No unified datasets found. Skipping retrain.")
                    continue
                big = pd.concat(dfs, ignore_index=True)
                feature_cols = [
                    "time_diff_min",
                    "price_change_pct",
                    "position_change",
                    "consecutive_wins",
                    "consecutive_losses",
                    "win_streak",
                    "loss_streak",
                ]
                avail = [c for c in feature_cols if c in big.columns]
                if not avail or "emotion" not in big.columns:
                    logger.info("Insufficient columns for retrain. Skipping.")
                    continue
                X = big[avail].fillna(0)
                y = big["emotion"].fillna("neutral")
                enc = LabelEncoder()
                y_enc = enc.fit_transform(y)
                new_model = RandomForestClassifier(n_estimators=200, random_state=42)
                new_model.fit(X, y_enc)
                # Swap in-memory safely
                async with model_lock:
                    joblib.dump(new_model, MODEL_PATH)
                    joblib.dump(enc, ENCODER_PATH)
                    global model, le
                    model, le = new_model, enc
                logger.info("✅ Retrained model and updated in memory")
            except Exception as e:
                logger.exception(f"Retrainer error: {e}")

    task = asyncio.create_task(background_retrainer())
    try:
        yield
    finally:
        task.cancel()
        logger.info("Shutting down backend")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan, title="Trading Psychology Backend", version="1.0")

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/dashboard/{user_id}/emotional_trends")
def emotional_trends(user_id: int):
    rows = moods.get(user_id, [])
    return {"count": len(rows), "rows": rows}


@app.post("/api/connect_wallet")
async def connect_wallet(request: ConnectWalletRequest):
    try:
        if not validate_wallet_address(request.walletType, request.walletAddress):
            raise HTTPException(status_code=400, detail="Invalid wallet address")
        
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_id FROM users WHERE wallet_address = ?",
            (request.walletAddress,)
        )
        existing_user = cursor.fetchone()
        
        if existing_user:
            user_id = existing_user[0]
        else:
            user_id = f"user_{hash(str(datetime.utcnow()) + request.walletAddress)}"
            cursor.execute(
                """
                INSERT INTO users (user_id, balance, wallet_address)
                VALUES (?, ?, ?)
                """,
                (user_id, 10000.0, request.walletAddress)
            )
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "userId": user_id,
            "walletAddress": request.walletAddress,
            "walletType": request.walletType
        }
    except Exception as e:
        logger.error(f"Error connecting wallet {request.walletAddress}: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect wallet")

@app.get("/performance-metrics", response_model=List[PerformanceMetric])
async def get_performance_metrics():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, metric, value, change, isPositive, icon FROM performance_metrics")
    metrics = [{"id": row[0], "metric": row[1], "value": row[2], "change": row[3], "isPositive": bool(row[4]), "icon": row[5]} for row in cursor.fetchall()]
    conn.close()
    return metrics

@app.get("/performance-distribution", response_model=PerformanceDistribution)
async def get_performance_distribution():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT profit, loss, breakEven FROM performance_distribution WHERE id = '1'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"profit": 0, "loss": 0, "breakEven": 0}
    return {"profit": row[0], "loss": row[1], "breakEven": row[2]}

@app.post("/mood", response_model=MoodEntry)
async def log_mood(entry: MoodEntry):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    mood_id = str(hash(f"{entry.mood}{datetime.utcnow()}"))
    day = (datetime.utcnow()).strftime("%A")
    cursor.execute("INSERT INTO mood_entries (id, day, mood, emoji, note) VALUES (?, ?, ?, ?, ?)",
                   (mood_id, day, entry.mood, entry.emoji, entry.note))
    conn.commit()
    conn.close()
    return {"id": mood_id, "day": day, "mood": entry.mood, "emoji": entry.emoji, "note": entry.note}

@app.get("/mood", response_model=List[MoodEntry])
async def get_mood_entries():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, day, mood, emoji, note FROM mood_entries")
    entries = [{"id": row[0], "day": row[1], "mood": row[2], "emoji": row[3], "note": row[4]} for row in cursor.fetchall()]
    conn.close()
    return entries

@app.get("/emotional-triggers", response_model=EmotionalTriggerData)
async def get_emotional_triggers():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT icon, label, value, color, description, greed, confidence, fear, revenge, fomo, emotionalScore FROM emotional_triggers")
    rows = cursor.fetchall()
    metrics = [
        {
            "icon": row[0],
            "label": row[1],
            "value": row[2],
            "color": row[3],
            "description": row[4],
        } for row in rows
    ]
    radar_data = {
        "greed": rows[0][5] if rows else 20.0,
        "confidence": rows[0][6] if rows else 75.0,
        "fear": rows[0][7] if rows else 30.0,
        "revenge": rows[0][8] if rows else 15.0,
        "fomo": rows[0][9] if rows else 25.0,
    }
    emotional_score = rows[0][10] if rows else 65.0
    conn.close()
    return {
        "metrics": metrics,
        "radarData": radar_data,
        "emotionalScore": emotional_score,
    }

@app.get("/ai-insights", response_model=Dict[str, List[AIInsight] | List[Recommendation]])
async def get_ai_insights():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, type, title, description, icon, color, bgColor FROM ai_insights")
    insights = [{"id": row[0], "type": row[1], "title": row[2], "description": row[3], "icon": row[4], "color": row[5], "bgColor": row[6]} for row in cursor.fetchall()]
    cursor.execute("SELECT id, text FROM recommendations")
    recommendations = [{"id": row[0], "text": row[1]} for row in cursor.fetchall()]
    conn.close()
    return {"insights": insights, "recommendations": recommendations}

@app.get("/feedback", response_model=List[FeedbackItem])
async def get_feedback():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, type, icon, title, message, priority, bgColor, iconColor FROM feedback")
    feedback = [
        {
            "id": row[0],
            "type": row[1],
            "icon": row[2],
            "title": row[3],
            "message": row[4],
            "priority": row[5],
            "bgColor": row[6],
            "iconColor": row[7],
        } for row in cursor.fetchall()
    ]
    conn.close()
    return feedback

@app.get("/recent-activities", response_model=List[RecentActivity])
async def get_recent_activities():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, type, title, description, timestamp, icon, color, value FROM recent_activities")
    activities = [
        {
            "id": row[0],
            "type": row[1],
            "title": row[2],
            "description": row[3],
            "timestamp": row[4],
            "icon": row[5],
            "color": row[6],
            "value": row[7],
        } for row in cursor.fetchall()
    ]
    conn.close()
    return activities

@app.post("/journal/")
def post_journal(payload: JournalEntry):
    journal.setdefault(payload.user_id, []).append({
        "entry": payload.entry,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {"status": "ok"}


@app.get("/chart-data", response_model=List[ChartDataPoint])
async def get_chart_data():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, value FROM chart_data")
    data = [{"id": row[0], "timestamp": row[1], "value": row[2]} for row in cursor.fetchall()]
    conn.close()
    return data

@app.get("/emotional-gauge", response_model=EmotionalGaugeData)
async def get_emotional_gauge():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT emotionalScore, maxScore, disciplineScore, disciplineColor, patienceScore, patienceColor, riskControlScore, riskControlColor FROM emotional_gauge WHERE id = '1'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {
            "emotionalScore": 7.2,
            "maxScore": 10.0,
            "breakdown": {
                "discipline": {"score": 8.5, "color": "bg-green-500"},
                "patience": {"score": 6.5, "color": "bg-yellow-500"},
                "riskControl": {"score": 7.0, "color": "bg-orange-500"},
            },
        }
    return {
        "emotionalScore": row[0],
        "maxScore": row[1],
        "breakdown": {
            "discipline": {"score": row[2], "color": row[3]},
            "patience": {"score": row[4], "color": row[5]},
            "riskControl": {"score": row[6], "color": row[7]},
        },
    }

@app.get("/emotional-score-gauge", response_model=EmotionalScoreGaugeData)
async def get_emotional_score_gauge():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT score, fearPercentage, fearColor, greedPercentage, greedColor, confidencePercentage, confidenceColor FROM emotional_score_gauge WHERE id = '1'")
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {
            "score": 76.0,
            "breakdown": {
                "fear": {"percentage": 25.0, "color": "bg-destructive"},
                "greed": {"percentage": 40.0, "color": "bg-warning"},
                "confidence": {"percentage": 76.0, "color": "bg-success"},
            },
        }
    return {
        "score": row[0],
        "breakdown": {
            "fear": {"percentage": row[1], "color": row[2]},
            "greed": {"percentage": row[3], "color": row[4]},
            "confidence": {"percentage": row[5], "color": row[6]},
        },
    }

@app.post("/api/place_order")
async def place_order(trade: TradeRequest):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    trade_id = str(hash(f"{trade.userId}{datetime.utcnow()}"))
    timestamp = datetime.utcnow().isoformat()
    wallet = f"0x{trade.userId}..."
    
    # Fetch current market price from Binance
    try:
        global market_data
        async with websockets.connect(f"wss://stream.binance.com:9443/ws/{trade.inToken}{trade.outToken.lower()}@trade") as ws:
            data = await ws.recv()
            market_data = json.loads(data)
            entry_price = float(market_data['p'])
    except Exception as e:
        logger.error(f"Failed to fetch market price: {e}")
        entry_price = trade.volumeUsd / trade.amountIn  # Fallback
    
    trade_data = {
        "id": trade_id,
        "user_id": trade.userId,
        "wallet": wallet,
        "in_token": trade.inToken,
        "out_token": trade.outToken,
        "amount_in": trade.amountIn,
        "amount_out": trade.amountOut,
        "volume_usd": trade.volumeUsd,
        "timestamp": timestamp,
        "emotion": "neutral",
        "trigger_details": None,
        "entry_price": entry_price,
        "exit_price": trade.volumeUsd / trade.amountOut,
        "pnl": (trade.volumeUsd / trade.amountOut - entry_price) * trade.amountOut * 0.997
    }
    
    # Engineer features and detect emotion
    cursor.execute(
        """
        SELECT id, user_id, wallet, in_token, out_token, amount_in, amount_out, volume_usd, timestamp, emotion, trigger_details, entry_price, exit_price, pnl
        FROM trades WHERE user_id = ? ORDER BY timestamp DESC
        """,
        (trade.userId,)
    )
    trades_data = [
        {
            "id": row[0],
            "user_id": row[1],
            "wallet": row[2],
            "in_token": row[3],
            "out_token": row[4],
            "amount_in": row[5],
            "amount_out": row[6],
            "volume_usd": row[7],
            "timestamp": row[8],
            "emotion": row[9],
            "trigger_details": row[10],
            "entry_price": row[11],
            "exit_price": row[12],
            "pnl": row[13]
        } for row in cursor.fetchall()
    ]
    trades_data.append(trade_data)
    df = pd.DataFrame(trades_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = engineer_features(df)
    df = detect_emotion(df)
    
    # Update emotion and trigger_details
    trade_data["emotion"] = df.iloc[-1]["emotion"]
    trade_data["trigger_details"] = df.iloc[-1]["trigger_details"]
    trade_data["entry_price"] = df.iloc[-1]["entry_price"] or entry_price
    trade_data["exit_price"] = df.iloc[-1]["exit_price"]
    trade_data["pnl"] = df.iloc[-1]["pnl"]
    
    # Save to database
    cursor.execute(
        """
        INSERT INTO trades (id, user_id, wallet, in_token, out_token, amount_in, amount_out, volume_usd, timestamp, emotion, trigger_details, entry_price, exit_price, pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_id,
            trade.userId,
            wallet,
            trade.inToken,
            trade.outToken,
            trade.amountIn,
            trade.amountOut,
            trade.volumeUsd,
            timestamp,
            trade_data["emotion"],
            trade_data["trigger_details"],
            trade_data["entry_price"],
            trade_data["exit_price"],
            trade_data["pnl"]
        )
    )
    conn.commit()
    
    # Train ML model
    ml_df = prepare_ml_data(df)
    if not ml_df.empty:
        X = ml_df.drop(columns=['emotion']).fillna(0)
        y = le.transform(ml_df['emotion'])
        model.fit(X, y)
        joblib.dump(model, 'emotion_predictor.pkl')
        joblib.dump(le, 'label_encoder.pkl')
    
    # Save feature vector
    features = df[['time_diff', 'price_change_pct', 'position_change', 'consecutive_wins', 'consecutive_losses', 'win_streak', 'loss_streak']].to_numpy()
    joblib.dump(features, f'user_{trade.userId}_features.pkl')
    
    # Predict next emotion
    predicted_emotion = "neutral"
    warning = {"insight": "No insight", "warning": "No warning", "recommendation": "None", "advice": "No advice"}
    
    # Calculate P&L and fees
    entry_price = trade.price if trade.orderType in ["limit", "stop"] else trade.volumeUsd / trade.amountIn
    fee = trade.volumeUsd * 0.001  # 0.1% fee
    trade_id = f"trade_{hash(str(datetime.utcnow()) + trade.userId)}"
    
    cursor.execute(
        """
        INSERT INTO trades (id, user_id, in_token, out_token, amount_in, amount_out, volume_usd, entry_price, fee, emotion, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trade_id, trade.userId, trade.inToken, trade.outToken, trade.amountIn, trade.amountOut, 
            trade.volumeUsd, entry_price, fee, predicted_emotion, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    if len(df) >= 4:
        predicted_emotion, _ = predict_next_emotion(df, model, le)
        warning = await get_emotion_warning(df, predicted_emotion)
    
    conn.close()

    return JSONResponse({
        "status": "success",
        "trade_id": trade_id,
        "predicted_emotion": predicted_emotion,
        "warning": warning
    })

@app.get("/api/market_data/{symbol}")
def get_market_symbol(symbol: str):
    if symbol not in market_data:
        raise HTTPException(404, "Symbol not found")
    return market_data[symbol][-1]

# ------------------ Unified Emotions ------------------
@app.get("/api/emotions/{user_id}")
async def unified_emotions(user_id: int):
    """Merge spot + futures, predict next emotion, call OpenAI, save unified dataset."""
    # Spot
    spot_df = None
    if user_id in UserTrade and UserTrade[user_id]:
        spot_df = pd.DataFrame(UserTrade[user_id])
        spot_df["timestamp"] = pd.to_datetime(spot_df["timestamp"])  # ensure datetime
        spot_df = engineer_features(spot_df)
        spot_df = detect_emotion(spot_df)
        spot_df["source"] = "spot"

    
        fut_positions = ["user_id", "market_data"] # and other relevant data that coud build up a DF
        if fut_positions:
            futures_df = pd.DataFrame(fut_positions)
            futures_df["source"] = "futures"
    

    dfs = [d for d in [spot_df, futures_df] if d is not None and not d.empty]
    if not dfs:
        return JSONResponse({"status": "no_data", "emotion": "neutral"})

    combined_df = pd.concat(dfs, ignore_index=True)

    # Save unified dataset (thread-safe)
    save_path = UNIFIED_FILE_TPL.format(uid=user_id)
    try:
        async with save_lock:
            joblib.dump(combined_df, save_path)
    except Exception as e:
        logger.warning(f"Couldn't save unified file: {e}")

    # Predict next emotion using current model
    predicted_emotion = "neutral"
    try:
        async with model_lock:
            if model is None or le is None:
                raise RuntimeError("Model not loaded")
            # Build minimal features from combined tail
            tmp = engineer_features(combined_df)
            tmp = detect_emotion(tmp)
            ml_df = prepare_ml_data(tmp)
            if not ml_df.empty:
                feature_cols = [
                    "time_diff_min",
                    "price_change_pct",
                    "position_change",
                    "consecutive_wins",
                    "consecutive_losses",
                    "win_streak",
                    "loss_streak",
                ]
                X = ml_df[feature_cols].fillna(0)
                y_pred = model.predict(X)
                predicted_emotion = str(le.inverse_transform([int(y_pred[-1])])[0])
    except Exception as e:
        logger.warning(f"Prediction failed: {e}")

    warning = call_openai_warning(combined_df, predicted_emotion)

    return JSONResponse(
        {
            "status": "success",
            "predicted_emotion": predicted_emotion,
            "warning": warning,
            "sources": combined_df["source"].value_counts().to_dict(),
            "saved_file": save_path,
        }
    )

@app.get("/api/balance")
async def get_balance(user_id: str):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {"user_id": user_id, "balance": float(result[0])}
        else:
            # Mock default balance for new users
            return {"user_id": user_id, "balance": 10000.0}
    except Exception as e:
        logger.error(f"Error fetching balance for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch balance")

@app.get("/api/user_trades", response_model=List[UserTrade])
async def get_user_trades(user_id: str, symbol: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, user_id, wallet, in_token, out_token, amount_in, amount_out, volume_usd, timestamp, emotion, trigger_details, entry_price, exit_price, pnl
        FROM trades WHERE user_id = ? AND in_token || out_token = ? ORDER BY timestamp DESC
        """,
        (user_id, symbol)
    )
    trades = [
        {
            "id": row[0],
            "user_id": row[1],
            "wallet": row[2],
            "in_token": row[3],
            "out_token": row[4],
            "amount_in": row[5],
            "amount_out": row[6],
            "volume_usd": row[7],
            "timestamp": row[8],
            "emotion": row[9],
            "trigger_details": row[10],
            "entry_price": row[11],
            "exit_price": row[12],
            "pnl": row[13]
        } for row in cursor.fetchall()
    ]
    conn.close()
    return [
            {
                "id": t[0],
                "in_token": t[1],
                "out_token": t[2],
                "amount_in": t[3],
                "entry_price": t[4],
                "timestamp": t[5],
                "pnl": t[6] if t[6] is not None else 0,
                "emotion": t[7]
            }
            for t in trades
        ]

@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, user_id, wallet, in_token, out_token, amount_in, amount_out, volume_usd, timestamp, emotion, trigger_details, entry_price, exit_price, pnl
        FROM trades WHERE user_id = ? ORDER BY timestamp DESC
        """,
        (user_id,)
    )
    trades_data = [
        {
            "id": row[0],
            "user_id": row[1],
            "wallet": row[2],
            "in_token": row[3],
            "out_token": row[4],
            "amount_in": row[5],
            "amount_out": row[6],
            "volume_usd": row[7],
            "timestamp": row[8],
            "emotion": row[9],
            "trigger_details": row[10],
            "entry_price": row[11],
            "exit_price": row[12],
            "pnl": row[13]
        } for row in cursor.fetchall()
    ]
    conn.close()
    
    if not trades_data or len(trades_data) < 4:
        return JSONResponse([])
    
    df = pd.DataFrame(trades_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = engineer_features(df)
    df = detect_emotion(df)
    predicted_emotion, _ = predict_next_emotion(df, model, le)
    warning = await get_emotion_warning(df, predicted_emotion)
    
    recommendations = [
        {"timestamp": datetime.utcnow().isoformat(), "message": f"Warning: {warning['warning']}", "severity": "warning"},
        {"timestamp": datetime.utcnow().isoformat(), "message": f"Recommendation: {warning['recommendation']}", "severity": "info"}
    ]
    return JSONResponse(recommendations)

@app.websocket("/ws/trades/{symbol}")
async def websocket_trades(websocket: WebSocket, symbol: str):
    await websocket.accept()
    try:
        try:
            symbols = market_data[symbol]
        except Exception as e:
            symbols = ["btcusdt", "ethusdt", "solusdt", "adausdt", "maticusdt", "icpusdt", "dotusdt"]
            logger.warning(f"Symbol {symbol} not found, defaulting to {symbols}: {e}")
        ws_url = f"wss://stream.binance.com:9443/ws/{'@trade/'.join(symbols)}@trade"
        async with websockets.connect(ws_url) as binance_ws:
            while True:
                data = await binance_ws.recv()
                trade_data = json.loads(data)
                trade = {
                    "id": trade_data['t'],
                    "symbol": trade_data['s'],
                    "type": "LONG" if trade_data['m'] else "SHORT",
                    "entry": float(trade_data['p']),
                    "exit": float(trade_data['p']),
                    "pnl": 0.0,
                    "emotion": "neutral",
                    "confidence": 50,
                    "timestamp": datetime.fromtimestamp(trade_data['T'] / 1000).isoformat(),
                }
                await websocket.send_text(json.dumps(trade))
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO trades (id, user_id, wallet, in_token, out_token, amount_in, amount_out, volume_usd, timestamp, emotion, trigger_details, entry_price, exit_price, pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade["id"],
                        "market",
                        "0xmarket...",
                        trade["symbol"].replace("USDT", ""),
                        "USDT",
                        trade_data['q'],
                        trade_data['q'],
                        float(trade_data['q']) * float(trade_data['p']),
                        trade["timestamp"],
                        trade["emotion"],
                        None,
                        trade["entry"],
                        trade["exit"],
                        trade["pnl"]
                    )
                )
                conn.commit()
                conn.close()
                await asyncio.sleep(0.1)  # Prevent overload
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=4001)


@app.get("/api/ticker")
async def get_ticker(symbol: str):
    try:
        async with websockets.connect(f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker") as ws:
            data = await ws.recv()
            ticker_data = json.loads(data)
            return {
                "symbol": ticker_data['s'],
                "lastPrice": float(ticker_data['c']),
                "priceChangePercent": float(ticker_data['P']),
            }
    except Exception as e:
        logger.error(f"Error fetching ticker for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch ticker data")

@app.get("/api/klines")
async def get_klines(symbol: str, interval: str, limit: int = 30):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            )
            res.raise_for_status()
            return [
                {
                    "t": item[0],  # timestamp
                    "o": float(item[1]),  # open
                    "h": float(item[2]),  # high
                    "l": float(item[3]),  # low
                    "c": float(item[4]),  # close
                }
                for item in res.json()
            ]
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch kline data")
    
# New endpoints
@app.post("/api/reset_session")
async def reset_session(user_id: str):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Session reset for user {user_id}"}
    except Exception as e:
        logger.error(f"Error resetting session for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset session")

@app.post("/api/save_strategy")
async def save_strategy(user_id: str, strategy: dict):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO strategies (user_id, strategy_data, timestamp) VALUES (?, ?, ?)",
            (user_id, json.dumps(strategy), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Strategy saved for user {user_id}"}
    except Exception as e:
        logger.error(f"Error saving strategy for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save strategy")

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "message": "Backend is running"}

@app.post("/api/emotions")
async def save_emotions(emotion: EmotionRequest):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO emotions (user_id, current_emotion, confidence, fear, excitement, stress, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (emotion.userId, emotion.currentEmotion, emotion.confidence, emotion.fear, 
             emotion.excitement, emotion.stress, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Emotions saved for user {emotion.userId}"}
    except Exception as e:
        logger.error(f"Error saving emotions for {emotion.userId}: {e}")
        raise HTTPException(status_code=500, detail="Failed to save emotions")

@app.get("/api/emotions")
async def get_emotions(user_id: str):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT current_emotion, confidence, fear, excitement, stress FROM emotions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
            (user_id,)
        )
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                "currentEmotion": result[0],
                "confidence": result[1],
                "fear": result[2],
                "excitement": result[3],
                "stress": result[4]
            }
        return {
            "currentEmotion": "neutral",
            "confidence": 7,
            "fear": 3,
            "excitement": 5,
            "stress": 4
        }
    except Exception as e:
        logger.error(f"Error fetching emotions for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch emotions")

@app.post("/api/emotions/reset")
async def reset_emotions(user_id: str):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute("DELETE FROM emotions WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        return {"status": "success", "message": f"Emotions reset for user {user_id}"}
    except Exception as e:
        logger.error(f"Error resetting emotions for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset emotions")

import numpy as np
@app.get("/api/performance")
async def get_performance(user_id: str, symbol: str = "ALL", mode: str = "live"):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        if symbol == "ALL":
            cursor.execute(
                """
                SELECT in_token, out_token, amount_in, amount_out, volume_usd, entry_price, fee, emotion, timestamp
                FROM trades WHERE user_id = ? AND mode = ?
                """,
                (user_id, mode)
            )
        else:
            cursor.execute(
                """
                SELECT in_token, out_token, amount_in, amount_out, volume_usd, entry_price, fee, emotion, timestamp
                FROM trades WHERE user_id = ? AND mode = ? AND (in_token || out_token) = ?
                """,
                (user_id, mode, symbol)
            )
        trades = cursor.fetchall()
        cursor.execute(
            "SELECT confidence, stress FROM emotions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
            (user_id,)
        )
        emotion_result = cursor.fetchone()
        conn.close()

        total_trades = len(trades)
        total_pnl = 0
        winning_trades = 0
        total_volume = 0
        hold_times = []
        pair_pnl = {}
        weekly_pnl = 0
        monthly_pnl = 0
        emotion_pnl = {}
        trade_pnls = []
        now = datetime.utcnow()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        for trade in trades:
            in_token, out_token, amount_in, amount_out, volume_usd, entry_price, fee, emotion, timestamp = trade
            pair = f"{in_token}/{out_token}"
            async with httpx.AsyncClient() as client:
                res = await client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={in_token}{out_token}")
                current_price = float(res.json().get("price", entry_price))
            trade_pnl = (current_price - entry_price) * amount_in - fee
            trade_pnls.append(trade_pnl)
            total_pnl += trade_pnl
            total_volume += volume_usd
            winning_trades += 1 if trade_pnl > 0 else 0
            pair_pnl[pair] = pair_pnl.get(pair, 0) + trade_pnl
            trade_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if trade_time >= week_ago:
                weekly_pnl += trade_pnl
            if trade_time >= month_ago:
                monthly_pnl += trade_pnl
            hold_times.append(4 * 3600 + 23 * 60)
            emotion_pnl[emotion] = emotion_pnl.get(emotion, 0) + trade_pnl

        total_pnl_percent = (total_pnl / total_volume * 100) if total_volume > 0 else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_hold_time = sum(hold_times) / len(hold_times) / 3600 if hold_times else 4.3833
        emotional_stability = (10 - (emotion_result[1] if emotion_result else 4) + (emotion_result[0] if emotion_result else 7)) / 2 if emotion_result else 7.2
        best_pair = max(pair_pnl.items(), key=lambda x: x[1], default=("BTC/USDT", 0))[0]
        worst_pair = min(pair_pnl.items(), key=lambda x: x[1], default=("SOL/USDT", 0))[0]
        best_pair_percent = pair_pnl.get(best_pair, 0) / total_volume * 100 if total_volume > 0 else 23.4
        worst_pair_percent = pair_pnl.get(worst_pair, 0) / total_volume * 100 if total_volume > 0 else -8.7
        best_emotion = max(emotion_pnl.items(), key=lambda x: x[1], default=("Confident", 0))[0]
        worst_emotion = min(emotion_pnl.items(), key=lambda x: x[1], default=("Fearful", 0))[0]
        improvement_area = "Reduce anxiety trades" if worst_emotion in ["anxious", "fearful", "frustrated"] else "Maintain confidence"
        best_trade = max(trade_pnls, default=0)
        worst_trade = min(trade_pnls, default=0)
        avg_trade = sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0
        daily_pnl = sum(trade_pnl for trade in trades if datetime.fromisoformat(trade[8].replace("Z", "+00:00")) >= now - timedelta(days=1))
        daily_change = (daily_pnl / total_volume * 100) if total_volume > 0 else 0
        sharpe_ratio = (np.mean(trade_pnls) / np.std(trade_pnls)) * np.sqrt(365) if trade_pnls and np.std(trade_pnls) != 0 else 2.34

        return {
            "totalPnL": round(total_pnl, 2),
            "totalPnLPercent": round(total_pnl_percent, 2),
            "winRate": round(win_rate, 1),
            "totalTrades": total_trades,
            "avgHoldTime": f"{int(avg_hold_time)}h {int((avg_hold_time % 1) * 60)}m",
            "emotionalStability": round(emotional_stability, 1),
            "bestPair": best_pair,
            "worstPair": worst_pair,
            "weeklyPnL": round(weekly_pnl, 2),
            "monthlyPnL": round(monthly_pnl, 2),
            "bestPairPercent": round(best_pair_percent, 1),
            "worstPairPercent": round(worst_pair_percent, 1),
            "bestEmotion": best_emotion,
            "worstEmotion": worst_emotion,
            "improvementArea": improvement_area,
            "bestTrade": round(best_trade, 2),
            "worstTrade": round(worst_trade, 2),
            "avgTrade": round(avg_trade, 2),
            "dailyChange": round(daily_change, 2),
            "sharpeRatio": round(sharpe_ratio, 2)
        }
    except Exception as e:
        logger.error(f"Error fetching performance for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch performance data")

@app.post("/api/feedback")
async def get_feedback(feedback: FeedbackRequest):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, type, message, accuracy, timestamp, priority
            FROM feedback WHERE user_id = ? AND (symbol = ? OR symbol = 'ALL') AND (mode = ? OR mode = 'all')
            ORDER BY timestamp DESC LIMIT 5
            """,
            (feedback.userId, feedback.symbol, feedback.mode)
        )
        feedback_data = cursor.fetchall()

        cursor.execute("SELECT COUNT(*) FROM trades WHERE user_id = ? AND mode = ?", (feedback.userId, feedback.mode))
        trade_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM emotions WHERE user_id = ?", (feedback.userId,))
        emotion_count = cursor.fetchone()[0]
        data_points = trade_count + emotion_count
        pattern_status = "Active" if trade_count > 10 else "Training"
        emotion_status = "Active" if emotion_count > 5 else "Training"

        conn.close()
        return {
            "insights": [
                {
                    "id": f[0],
                    "type": f[1],
                    "title": f[1].capitalize() + " Insight",
                    "description": f[2],
                    "confidence": f[3],
                    "timestamp": f[4],
                    "priority": f[5],
                    "action": {
                        "warning": "Take Break",
                        "opportunity": "View Setup",
                        "insight": "Learn More",
                        "strategy": "Adjust Size"
                    }.get(f[1], "Review"),
                    "icon": {
                        "warning": "AlertTriangle",
                        "opportunity": "TrendingUp",
                        "insight": "CheckCircle",
                        "strategy": "Target"
                    }.get(f[1], "Brain"),
                    "bgColor": {
                        "warning": "bg-warning/10",
                        "opportunity": "bg-success/10",
                        "insight": "bg-primary/10",
                        "strategy": "bg-accent/10"
                    }.get(f[1], "bg-muted/10"),
                    "color": {
                        "warning": "text-warning",
                        "opportunity": "text-success",
                        "insight": "text-primary",
                        "strategy": "text-accent"
                    }.get(f[1], "text-muted-foreground")
                }
                for f in feedback_data
            ],
            "recommendations": [
                {
                    "id": f[0],
                    "text": {
                        "warning": f"Take a break to reset your emotional state (triggered by {f[1]}).",
                        "opportunity": f"Review setup for {feedback.symbol} ({f[1]}).",
                        "insight": f"Learn more about your trading patterns ({f[1]}).",
                        "strategy": f"Adjust position size for {feedback.symbol} ({f[1]})."
                    }.get(f[1], f"Review your {f[1]} feedback.")
                }
                for f in feedback_data
            ],
            "ai_status": {
                "patternRecognition": pattern_status,
                "emotionalAnalysis": emotion_status,
                "dataPoints": data_points
            }
        }
    except Exception as e:
        logger.error(f"Error fetching feedback for {feedback.userId}, {feedback.symbol}, {feedback.mode}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch feedback")
    

@app.get("/api/performance_timeseries")
async def get_performance_timeseries(user_id: str, symbol: str = "ALL", mode: str = "live"):
    try:
        conn = sqlite3.connect("trading_app.db")
        cursor = conn.cursor()
        if symbol == "ALL":
            cursor.execute(
                """
                SELECT in_token, out_token, amount_in, entry_price, fee, timestamp, volume_usd
                FROM trades WHERE user_id = ? AND mode = ?
                ORDER BY timestamp
                """,
                (user_id, mode)
            )
        else:
            cursor.execute(
                """
                SELECT in_token, out_token, amount_in, entry_price, fee, timestamp, volume_usd
                FROM trades WHERE user_id = ? AND mode = ? AND (in_token || out_token) = ?
                ORDER BY timestamp
                """,
                (user_id, mode, symbol)
            )
        trades = cursor.fetchall()
        cursor.execute(
            """
            SELECT confidence, fear, excitement, stress, timestamp
            FROM emotions WHERE user_id = ?
            ORDER BY timestamp
            """,
            (user_id,)
        )
        emotions = cursor.fetchall()
        conn.close()

        now = datetime.utcnow()
        start_time = now - timedelta(days=1)
        buckets = []
        for i in range(0, 24, 4):
            bucket_start = start_time + timedelta(hours=i)
            bucket_end = bucket_start + timedelta(hours=4)
            bucket_trades = [
                t for t in trades
                if bucket_start <= datetime.fromisoformat(t[5].replace("Z", "+00:00")) < bucket_end
            ]
            bucket_emotions = [
                e for e in emotions
                if bucket_start <= datetime.fromisoformat(e[4].replace("Z", "+00:00")) < bucket_end
            ]
            bucket_pnl = 0
            bucket_volume = 0
            for trade in bucket_trades:
                in_token, out_token, amount_in, entry_price, fee = trade[0], trade[1], trade[2], trade[3], trade[4]
                async with httpx.AsyncClient() as client:
                    res = await client.get(f"https://api.binance.com/api/v3/ticker/price?symbol={in_token}{out_token}")
                    current_price = float(res.json().get("price", entry_price))
                bucket_pnl += (current_price - entry_price) * amount_in - fee
                bucket_volume += trade[6]
            bucket_emotional = sum((e[0] - e[1] + e[2] - e[3]) for e in bucket_emotions) / len(bucket_emotions) * 10 if bucket_emotions else 50
            buckets.append({
                "time": bucket_start.isoformat(),
                "pnl": round(bucket_pnl, 2),
                "emotional": round(bucket_emotional, 2),
                "volume": round(bucket_volume, 2)
            })

        return buckets
    except Exception as e:
        logger.error(f"Error fetching performance timeseries for {user_id}, {symbol}, {mode}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch performance timeseries")

# ---------------------------------------------------------
# Archetype Endpoints and APIs
# ---------------------------------------------------------
@app.post("/api/archetype/assign/{user_id}", response_model=ArchetypeResponse)
async def assign_archetype(user_id: str, days: int = 30):
    """
    Assign trading archetype based on user's trading patterns and emotions
    """
    try:
        # Get user trading data
        patterns = await analyze_trading_patterns(user_id, days)
        archetype_data = calculate_archetype(patterns)
        
        return ArchetypeResponse(
            user_id=user_id,
            archetype=archetype_data["archetype"],
            confidence=archetype_data["confidence"],
            traits=archetype_data["traits"],
            description=archetype_data["description"],
            recommendations=archetype_data["recommendations"]
        )
    except Exception as e:
        logger.error(f"Error assigning archetype for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to assign archetype")

async def analyze_trading_patterns(user_id: str, days: int) -> TradingPatterns:
    """Analyze user's trading patterns over specified period"""
    conn = sqlite3.connect(DATABASE)
    
    # Get trades within time period
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, amount_in, amount_out, volume_usd, pnl, emotion
        FROM trades 
        WHERE user_id = ? AND timestamp >= ?
        ORDER BY timestamp
    """, (user_id, start_date))
    
    trades = cursor.fetchall()
    
    if not trades:
        return TradingPatterns(
            avg_trade_frequency_min=0,
            win_rate=0,
            avg_hold_time_hours=0,
            risk_reward_ratio=0,
            emotional_volatility=0,
            consecutive_losses=0,
            fomo_score=0,
            greed_score=0,
            fear_score=0
        )
    
    df = pd.DataFrame(trades, columns=['timestamp', 'amount_in', 'amount_out', 'volume_usd', 'pnl', 'emotion'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_win'] = df['pnl'] > 0
    
    # Calculate patterns
    time_diffs = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
    avg_trade_frequency = time_diffs.mean() if len(time_diffs) > 1 else 0
    
    win_rate = df['is_win'].mean() * 100
    
    # Emotional analysis
    emotion_counts = df['emotion'].value_counts()
    total_trades = len(df)
    
    fomo_score = (emotion_counts.get('fomo', 0) / total_trades) * 100
    greed_score = (emotion_counts.get('greed', 0) / total_trades) * 100
    fear_score = (emotion_counts.get('fear', 0) / total_trades) * 100
    
    # Consecutive losses
    consecutive_losses = 0
    current_streak = 0
    for pnl in df['pnl']:
        if pnl < 0:
            current_streak += 1
            consecutive_losses = max(consecutive_losses, current_streak)
        else:
            current_streak = 0
    
    # Risk-reward ratio (simplified)
    winning_trades = df[df['is_win']]['pnl']
    losing_trades = df[~df['is_win']]['pnl']
    
    avg_win = winning_trades.mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 1
    risk_reward_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    
    # Emotional volatility (standard deviation of emotion scores)
    emotion_scores = {
        'neutral': 0, 'fomo': 80, 'greed': 70, 
        'fear': -60, 'revenge': -40
    }
    df['emotion_score'] = df['emotion'].map(emotion_scores)
    emotional_volatility = df['emotion_score'].std() if len(df) > 1 else 0
    
    conn.close()
    
    return TradingPatterns(
        avg_trade_frequency_min=avg_trade_frequency,
        win_rate=win_rate,
        avg_hold_time_hours=24,  # Simplified
        risk_reward_ratio=risk_reward_ratio,
        emotional_volatility=emotional_volatility,
        consecutive_losses=consecutive_losses,
        fomo_score=fomo_score,
        greed_score=greed_score,
        fear_score=fear_score
    )

def calculate_archetype(patterns: TradingPatterns) -> Dict:
    """Calculate archetype based on trading patterns"""
    
    # Score calculations
    fomo_risk = (patterns.fomo_score * 0.4 + 
                 (100 - patterns.avg_trade_frequency_min) * 0.3 + 
                 patterns.emotional_volatility * 0.3)
    
    greed_risk = (patterns.greed_score * 0.5 + 
                  patterns.risk_reward_ratio * 0.3 + 
                  (100 - patterns.win_rate) * 0.2)
    
    fear_risk = (patterns.fear_score * 0.6 + 
                 patterns.consecutive_losses * 0.4)
    
    rationality_score = (patterns.win_rate * 0.3 + 
                        min(patterns.risk_reward_ratio, 3) * 0.3 + 
                        (100 - patterns.emotional_volatility) * 0.4)
    
    # Archetype assignment
    archetype_scores = {
        Archetype.FOMO_APE: fomo_risk,
        Archetype.GREEDY_BULL: greed_risk,
        Archetype.FEARFUL_WHALE: fear_risk,
        Archetype.RATIONAL_TRADER: rationality_score,
        Archetype.REVENGE_TRADER: patterns.consecutive_losses * 20,
        Archetype.PATIENT_HODLER: (patterns.avg_trade_frequency_min * 0.5 + 
                                  patterns.win_rate * 0.5)
    }
    
    assigned_archetype = max(archetype_scores.items(), key=lambda x: x[1])[0]
    confidence = min(archetype_scores[assigned_archetype] / 100, 1.0)
    
    # Traits breakdown
    traits = {
        "fomo_tendency": patterns.fomo_score / 100,
        "greed_tendency": patterns.greed_score / 100,
        "fear_tendency": patterns.fear_score / 100,
        "rationality": rationality_score / 100,
        "patience": (patterns.avg_trade_frequency_min / 1440),  # Convert minutes to days
        "risk_tolerance": min(patterns.risk_reward_ratio / 3, 1.0)
    }
    
    # Descriptions and recommendations /// sub with AI recommendation and use dis as fallback
    archetype_info = {
        Archetype.FOMO_APE: {
            "description": "Tends to chase pumps and enter positions based on market hype",
            "recommendations": [
                "Set strict entry/exit criteria before trading",
                "Avoid trading during high volatility periods",
                "Use dollar-cost averaging instead of lump sum investments"
            ]
        },
        Archetype.GREEDY_BULL: {
            "description": "Overtrades during wins and holds losing positions too long",
            "recommendations": [
                "Implement take-profit targets",
                "Use trailing stops to protect gains",
                "Take breaks after significant wins"
            ]
        },
        Archetype.FEARFUL_WHALE: {
            "description": "Exits positions too early and misses potential gains",
            "recommendations": [
                "Set predefined profit targets",
                "Use partial profit taking strategies",
                "Review trade journals to build confidence"
            ]
        },
        Archetype.RATIONAL_TRADER: {
            "description": "Makes calculated decisions based on analysis rather than emotion",
            "recommendations": [
                "Continue current risk management practices",
                "Consider scaling position sizes strategically",
                "Mentor other traders in emotional control"
            ]
        },
        Archetype.REVENGE_TRADER: {
            "description": "Trades to recover losses rather than based on market opportunities",
            "recommendations": [
                "Implement daily loss limits",
                "Take 24-hour breaks after significant losses",
                "Focus on process over outcomes"
            ]
        },
        Archetype.PATIENT_HODLER: {
            "description": "Prefers long-term holding over active trading",
            "recommendations": [
                "Consider strategic rebalancing",
                "Evaluate portfolio diversification regularly",
                "Monitor macro trends for exit opportunities"
            ]
        }
    }
    
    return {
        "archetype": assigned_archetype,
        "confidence": confidence,
        "traits": traits,
        "description": archetype_info[assigned_archetype]["description"],
        "recommendations": archetype_info[assigned_archetype]["recommendations"]
    }

# ---------------------------------------------------------
# Market trend Endpoints and APIs
# ---------------------------------------------------------
@app.post("/api/market/analyze/{user_id}", response_model=MarketAnalysisResponse)
async def analyze_market_trends(
    user_id: str, 
    request: MarketAnalysisRequest
):
    """
    Analyze market trends with RSI and breakout detection for multiple symbols
    """
    try:
        analysis_results = {}
        
        for symbol in request.symbols:
            try:
                # Try primary exchange first, fallback to DUNI
                if request.exchange == "solana":
                    data = await fetch_solana_data(symbol, request.timeframe)
                else:  # binance default
                    data = await fetch_binance_data(symbol, request.timeframe)
                
                if data.empty:
                    data = await fetch_duni_data(symbol, request.timeframe)
                
                indicators = calculate_technical_indicators(data, symbol)
                analysis_results[symbol] = indicators
                
            except Exception as e:
                logger.warning(f"Failed to analyze {symbol}: {e}")
                # Provide fallback analysis
                analysis_results[symbol] = TechnicalIndicators(
                    symbol=symbol,
                    rsi=50.0,
                    rsi_signal="neutral",
                    breakout_detected=False,
                    breakout_strength=0.0,
                    volume_spike=False,
                    trend_direction="neutral",
                    confidence=0.0
                )
        
        return MarketAnalysisResponse(
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            analysis=analysis_results
        )
        
    except Exception as e:
        logger.error(f"Market analysis failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Market analysis failed")

async def fetch_binance_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch OHLCV data from Binance"""
    try:
        async with httpx.AsyncClient() as client:
            # Convert timeframe to Binance format
            tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
            binance_tf = tf_map.get(timeframe, "1h")
            
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": binance_tf,
                "limit": 100
            }
            
            response = await client.get(url, params=params, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert to proper types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
    except Exception as e:
        logger.warning(f"Binance data fetch failed for {symbol}: {e}")
        return pd.DataFrame()

async def fetch_solana_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch Solana token data (simplified - would integrate with Solana DEX APIs)"""
    try:
        # This is a placeholder - in reality you'd use:
        # - Jupiter API for Solana token prices
        # - Raydium API for DEX data
        # - Birdeye for comprehensive Solana data
        
        async with httpx.AsyncClient() as client:
            # Using Birdeye as example for Solana data
            url = f"https://public-api.birdeye.so/public/history_price"
            params = {
                "address": symbol,  # This would be token mint address
                "type": timeframe,
                "time_from": int((datetime.utcnow() - timedelta(days=7)).timestamp()),
                "time_to": int(datetime.utcnow().timestamp())
            }
            
            headers = {
                "X-API-KEY": os.getenv("BIRDEYE_API_KEY", "")
            }
            
            response = await client.get(url, params=params, headers=headers, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                # Process Birdeye response format
                # This would need to be adapted based on actual API response
                return pd.DataFrame()
            else:
                return pd.DataFrame()
                
    except Exception as e:
        logger.warning(f"Solana data fetch failed for {symbol}: {e}")
        return pd.DataFrame()

async def fetch_duni_data(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fallback data provider (DUNI)"""
    try:
        # DUNI would be your custom fallback data source
        # This could be a cached database, alternative API, etc.
        async with httpx.AsyncClient() as client:
            url = f"https://api.duni.io/market/data"  # Example endpoint
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "limit": 100
            }
            
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                # Process DUNI response format
                return pd.DataFrame(data.get('candles', []))
            else:
                return pd.DataFrame()
                
    except Exception as e:
        logger.warning(f"DUNI data fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame, symbol: str) -> TechnicalIndicators:
    """Calculate RSI, breakout detection and other technical indicators"""
    if df.empty:
        return TechnicalIndicators(
            symbol=symbol,
            rsi=50.0,
            rsi_signal="neutral",
            breakout_detected=False,
            breakout_strength=0.0,
            volume_spike=False,
            trend_direction="neutral",
            confidence=0.0
        )
    
    # Calculate RSI
    rsi = calculate_rsi(df['close'])
    rsi_signal = "neutral"
    if rsi > 70:
        rsi_signal = "overbought"
    elif rsi < 30:
        rsi_signal = "oversold"
    
    # Breakout detection
    breakout_detected, breakout_strength = detect_breakout(df)
    
    # Volume analysis
    volume_spike = detect_volume_spike(df)
    
    # Trend direction
    trend_direction = determine_trend(df)
    
    # Support and resistance levels
    support, resistance = calculate_support_resistance(df)
    
    # Overall confidence based on data quality and indicator consistency
    confidence = calculate_analysis_confidence(df, rsi, breakout_detected)
    
    return TechnicalIndicators(
        symbol=symbol,
        rsi=round(rsi, 2),
        rsi_signal=rsi_signal,
        breakout_detected=breakout_detected,
        breakout_strength=round(breakout_strength, 2),
        volume_spike=volume_spike,
        trend_direction=trend_direction,
        support_level=support,
        resistance_level=resistance,
        confidence=round(confidence, 2)
    )

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate Relative Strength Index"""
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty else 50.0

def detect_breakout(df: pd.DataFrame, lookback_period: int = 20) -> tuple:
    """Detect price breakouts using volatility and resistance breaks"""
    if len(df) < lookback_period + 1:
        return False, 0.0
    
    current_close = df['close'].iloc[-1]
    current_high = df['high'].iloc[-1]
    
    # Calculate recent resistance level (recent high)
    recent_high = df['high'].rolling(window=lookback_period).max().iloc[-2]
    
    # Calculate average true range for volatility
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=lookback_period).mean().iloc[-1]
    
    # Breakout detection
    breakout_strength = 0.0
    breakout_detected = False
    
    if current_high > recent_high and atr > 0:
        # Strength based on how far above resistance and volume confirmation
        distance_above = (current_high - recent_high) / recent_high * 100
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(window=lookback_period).mean().iloc[-1]
        
        breakout_strength = min(distance_above * volume_ratio, 100.0)
        breakout_detected = breakout_strength > 5.0  # Minimum strength threshold
    
    return breakout_detected, breakout_strength

def detect_volume_spike(df: pd.DataFrame, period: int = 20) -> bool:
    """Detect unusual volume spikes"""
    if len(df) < period:
        return False
    
    current_volume = df['volume'].iloc[-1]
    avg_volume = df['volume'].rolling(window=period).mean().iloc[-2]
    
    return current_volume > avg_volume * 2.0  # 2x average volume

def determine_trend(df: pd.DataFrame) -> str:
    """Determine overall trend direction"""
    if len(df) < 10:
        return "neutral"
    
    # Simple moving average trend
    sma_short = df['close'].rolling(window=5).mean().iloc[-1]
    sma_long = df['close'].rolling(window=20).mean().iloc[-1]
    
    if sma_short > sma_long * 1.02:  # 2% above
        return "bullish"
    elif sma_short < sma_long * 0.98:  # 2% below
        return "bearish"
    else:
        return "neutral"

def calculate_support_resistance(df: pd.DataFrame) -> tuple:
    """Calculate basic support and resistance levels"""
    if len(df) < 20:
        return None, None
    
    # Simplified: use recent lows and highs
    support = df['low'].tail(20).min()
    resistance = df['high'].tail(20).max()
    
    return float(support), float(resistance)

def calculate_analysis_confidence(df: pd.DataFrame, rsi: float, breakout: bool) -> float:
    """Calculate confidence score for the analysis"""
    confidence_factors = []
    
    # Data quality factor
    data_points = len(df)
    data_quality = min(data_points / 50.0, 1.0)  # Normalize to 0-1
    confidence_factors.append(data_quality * 0.3)
    
    # RSI clarity factor
    rsi_clarity = 0.0
    if rsi < 30 or rsi > 70:
        rsi_clarity = 1.0  # Clear signal
    elif rsi < 40 or rsi > 60:
        rsi_clarity = 0.6  # Moderate signal
    else:
        rsi_clarity = 0.3  # Weak signal
    confidence_factors.append(rsi_clarity * 0.3)
    
    # Breakout clarity factor
    breakout_clarity = 1.0 if breakout else 0.5
    confidence_factors.append(breakout_clarity * 0.2)
    
    # Volume confirmation factor
    volume_trend = df['volume'].tail(5).mean() / df['volume'].mean()
    volume_factor = min(volume_trend, 1.0)
    confidence_factors.append(volume_factor * 0.2)
    
    return sum(confidence_factors)

# route
@app.get("/")
async def root():
    return {"message": "Welcome to the Trading App Backend! Visit localhost:5000/docs for API documentation."}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)