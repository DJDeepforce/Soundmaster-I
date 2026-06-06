from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import json
from datetime import datetime, timezone, timedelta
import numpy as np
import tempfile
import io
import time
import asyncio
import anthropic
import librosa
import pyloudnorm as pyln
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.enums import TA_CENTER
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import filetype

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

client_db = AsyncIOMotorClient(os.environ.get('MONGO_URL', ''))
db = client_db[os.environ.get('DB_NAME', '')]
anthropic_client = anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY', ''))

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
api_router = APIRouter(prefix="/api")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.aiff', '.aif', '.ogg')
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'changeme-please-set-in-env')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 jours
TIER_LIMITS = {"free": 2, "starter": 15, "pro": 50}

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")
limiter = Limiter(key_func=get_remote_address)

def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429,
        content={"detail": "Trop de requêtes. Réessayez dans 1 minute."})

# ==================== MODELS ====================

class FrequencyBand(BaseModel):
    range_hz: str
    energy_db: float
    peak_freq: float

class LoudnessAnalysis(BaseModel):
    rms_db: float
    peak_db: float
    lufs_integrated: float
    lufs_estimate: float
    dynamic_range_db: float
    crest_factor_db: float
    headroom_db: float
    true_peak_db: float

class StereoAnalysis(BaseModel):
    is_stereo: bool
    correlation: float
    width: float
    balance: float
    mid_energy_db: Optional[float] = None
    side_energy_db: Optional[float] = None
    phase_issues: Optional[bool] = None
    note: Optional[str] = None

class AnalysisResult(BaseModel):
    id: str
    filename: str
    duration: float
    sample_rate: int
    channels: int
    frequency_analysis: Dict[str, FrequencyBand]
    loudness_analysis: LoudnessAnalysis
    stereo_analysis: StereoAnalysis
    recommendations: List[str]
    overall_score: int
    status: str
    spectrogram_3d: Optional[Dict] = None

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None
    history: Optional[List[Dict]] = None

class ChatResponse(BaseModel):
    response: str

class UserCreate(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    user_id: str
    email: str
    tier: str
    analyses_used: int

# ==================== AUTH ====================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Token invalide")
    except JWTError:
        raise HTTPException(401, "Token invalide ou expiré")
    user = await db.users.find_one({"user_id": user_id}, {"_id": 0, "password": 0})
    if not user:
        raise HTTPException(401, "Utilisateur introuvable")
    # Reset mensuel du quota
    reset_at = user.get("analyses_reset_at")
    if not reset_at or (datetime.now(timezone.utc) - datetime.fromisoformat(reset_at)) > timedelta(days=30):
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"analyses_used": 0, "analyses_reset_at": datetime.now(timezone.utc).isoformat()}}
        )
        user["analyses_used"] = 0
    return user

@app.on_event("startup")
async def setup_indexes():
    await db.users.create_index("email", unique=True)
    await db.audio_analyses.create_index("user_id")
    await db.audio_analyses.create_index([("created_at", -1)])

# ==================== AUDIO FUNCTIONS ====================

def generate_3d_spectrogram(y, sr, max_time=100, max_freq=60):
    n_fft = 2048
    hop = max(512, len(y) // max_time)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop)), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= 20) & (freqs <= 20000)
    F, D = freqs[mask], D[mask, :]
    if len(F) > max_freq:
        idx = np.linspace(0, len(F)-1, max_freq, dtype=int)
        F, D = F[idx], D[idx, :]
    if D.shape[1] > max_time:
        idx = np.linspace(0, D.shape[1]-1, max_time, dtype=int)
        D = D[:, idx]
    D_norm = (D - D.min()) / (D.max() - D.min() + 1e-10)
    times = np.linspace(0, len(y)/sr, D.shape[1])
    return {"times": times.tolist(), "frequencies": F.tolist(), "amplitudes": D_norm.tolist(),
            "duration": len(y)/sr, "freq_min": float(F.min()), "freq_max": float(F.max())}

def analyze_frequency_bands(y, sr):
    bands = {'sub_bass':(20,60),'bass':(60,250),'low_mids':(250,500),
             'mids':(500,2000),'high_mids':(2000,4000),'highs':(4000,8000),'air':(8000,20000)}
    D = np.abs(librosa.stft(y, n_fft=4096))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    power = np.mean(D**2, axis=1)
    result = {}
    for name, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if len(idx) > 0:
            bp = power[idx]
            edb = 10 * np.log10(np.mean(bp) + 1e-10)
            pf = freqs[idx[np.argmax(bp)]]
        else:
            edb, pf = -60.0, (lo+hi)/2
        result[name] = FrequencyBand(range_hz=f"{lo}-{hi}", energy_db=round(edb,2), peak_freq=round(pf,1))
    return result

def analyze_loudness(y, sr):
    rms_db = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
    peak_db = 20 * np.log10(np.max(np.abs(y)) + 1e-10)
    headroom_db = -peak_db

    # Vrai LUFS ITU-R BS.1770-4
    try:
        meter = pyln.Meter(sr)
        y_lufs = y.astype(np.float64).reshape(-1, 1) if y.ndim == 1 else y.T.astype(np.float64)
        lufs = meter.integrated_loudness(y_lufs)
        lufs = rms_db - 0.691 if (np.isinf(lufs) or np.isnan(lufs)) else lufs
    except Exception:
        lufs = rms_db - 0.691

    # True Peak — cap to 30s before 4x upsample to avoid OOM on free-tier servers
    try:
        max_samples = sr * 30
        y_clip = y[:max_samples] if len(y) > max_samples else y
        y_up = librosa.resample(y_clip, orig_sr=sr, target_sr=sr * 4)
        true_peak_db = 20 * np.log10(np.max(np.abs(y_up)) + 1e-10)
    except Exception:
        true_peak_db = peak_db

    # Dynamic range
    fl = min(int(sr*0.4), max(256, len(y)//4))
    hl = max(64, fl//4)
    dr = 0.0
    if len(y) > fl:
        try:
            frames = librosa.util.frame(y, frame_length=fl, hop_length=hl)
            fdb = 20 * np.log10(np.sqrt(np.mean(frames**2, axis=0)) + 1e-10)
            valid = fdb[fdb > -60]
            if len(valid) > 1:
                dr = np.percentile(valid, 95) - np.percentile(valid, 5)
        except Exception:
            pass

    return LoudnessAnalysis(
        rms_db=round(rms_db,2), peak_db=round(peak_db,2),
        lufs_integrated=round(lufs,2), lufs_estimate=round(lufs,2),
        dynamic_range_db=round(dr,2), crest_factor_db=round(peak_db-rms_db,2),
        headroom_db=round(headroom_db,2), true_peak_db=round(true_peak_db,2)
    )

def analyze_stereo(y, sr):
    if y.ndim == 1:
        return StereoAnalysis(is_stereo=False, correlation=1.0, width=0.0, balance=0.0,
                              note="Fichier mono - pas d'analyse stéréo disponible")
    l, r = y[0], y[1]
    mid, side = (l+r)/2, (l-r)/2
    me, se = np.mean(mid**2), np.mean(side**2)
    corr = float(np.corrcoef(l, r)[0,1])
    if np.isnan(corr): corr = 1.0
    width = float(np.sqrt(se/(me+se))) if me > 0 else 0.0
    le, re = np.mean(l**2), np.mean(r**2)
    te = le + re
    bal = float((re-le)/te) if te > 0 else 0.0
    return StereoAnalysis(
        is_stereo=True, correlation=round(corr,3), width=round(width,3), balance=round(bal,3),
        mid_energy_db=round(10*np.log10(me+1e-10),2), side_energy_db=round(10*np.log10(se+1e-10),2),
        phase_issues=corr < 0.3
    )

def calculate_score(freq, loudness, stereo):
    score = 100
    if loudness.headroom_db < 1: score -= 15
    elif loudness.headroom_db < 3: score -= 5
    if loudness.true_peak_db > 0: score -= 10
    if loudness.dynamic_range_db < 5: score -= 10
    elif loudness.dynamic_range_db < 8: score -= 5
    if freq['bass'].energy_db > freq['mids'].energy_db + 6: score -= 10
    if freq['highs'].energy_db > freq['mids'].energy_db + 3: score -= 5
    if freq['highs'].energy_db < freq['mids'].energy_db - 12: score -= 5
    if stereo.is_stereo:
        if stereo.phase_issues: score -= 10
        if abs(stereo.balance) > 0.2: score -= 5
    return max(0, min(100, score))

async def get_recommendations(freq, loudness, stereo, filename, tier: str = "free"):
    # Tier free : recommandations statiques, pas d'appel Claude
    if tier == "free":
        recs = []
        if loudness.true_peak_db > 0:
            recs.append(f"True Peak à {loudness.true_peak_db} dBTP — clipping détecté. Baissez le master et relimitez.")
        if loudness.headroom_db < 1:
            recs.append("Headroom insuffisant. Réduisez le gain master de 2-3 dB.")
        if loudness.dynamic_range_db < 6:
            recs.append("Mix sur-compressé. Réduisez le ratio de compression sur le bus master.")
        if freq['bass'].energy_db > freq['mids'].energy_db + 6:
            recs.append("Basses trop présentes. Cut EQ vers 200-300Hz.")
        if stereo.phase_issues:
            recs.append("Problème de phase stéréo. Vérifiez la polarité de vos pistes.")
        if not recs:
            recs.append("Mix équilibré. Passez au plan Starter pour des recommandations IA détaillées.")
        return recs
    summary = f"""
Fichier: {filename}
LUFS intégré: {loudness.lufs_integrated} LUFS | True Peak: {loudness.true_peak_db} dBTP
RMS: {loudness.rms_db} dB | Peak: {loudness.peak_db} dB | Headroom: {loudness.headroom_db} dB
Plage dynamique: {loudness.dynamic_range_db} dB | Crest factor: {loudness.crest_factor_db} dB

SPECTRAL:
Sub Bass: {freq['sub_bass'].energy_db} dB | Bass: {freq['bass'].energy_db} dB
Low Mids: {freq['low_mids'].energy_db} dB | Mids: {freq['mids'].energy_db} dB
High Mids: {freq['high_mids'].energy_db} dB | Highs: {freq['highs'].energy_db} dB | Air: {freq['air'].energy_db} dB

STÉRÉO: {'Stéréo' if stereo.is_stereo else 'Mono'} | Corrélation: {stereo.correlation} {'⚠️ PHASE' if stereo.phase_issues else ''}
Largeur: {stereo.width} | Balance: {stereo.balance}

Références: Spotify -14 LUFS | Apple Music -16 LUFS | YouTube -14 LUFS | Club -6/-8 LUFS
"""
    try:
        def _call_claude():
            return anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1024,
                system="""Ingénieur du son professionnel expert mixage/mastering.
5-6 recommandations concrètes en français, une par ligne.
Mentionne les standards LUFS des plateformes si pertinent.
Sois précis et technique. Ne numérote pas.""",
                messages=[{"role": "user", "content": f"Recommandations pour ce mix:\n{summary}"}]
            )
        msg = await asyncio.to_thread(_call_claude)
        recs = [r.strip() for r in msg.content[0].text.strip().split('\n') if r.strip() and len(r.strip()) > 10]
        return recs[:6]
    except Exception as e:
        logger.error(f"Claude error: {e}")
        recs = []
        if loudness.true_peak_db > 0:
            recs.append(f"True Peak à {loudness.true_peak_db} dBTP — clipping détecté. Baissez le master et relimitez.")
        if loudness.headroom_db < 1:
            recs.append("Headroom insuffisant. Réduisez le gain master de 2-3 dB.")
        if loudness.dynamic_range_db < 6:
            recs.append("Mix sur-compressé. Réduisez le ratio de compression sur le bus master.")
        if freq['bass'].energy_db > freq['mids'].energy_db + 6:
            recs.append("Basses trop présentes. Cut EQ vers 200-300Hz.")
        if stereo.phase_issues:
            recs.append("Problème de phase stéréo. Vérifiez la polarité de vos pistes.")
        if not recs:
            recs.append("Mix équilibré. Continuez à affiner les détails.")
        return recs

# ==================== ROUTES ====================

@api_router.get("/")
async def root():
    return {"message": "SoundMaster API v2 — Powered by Claude"}

@api_router.post("/register", response_model=Token)
async def register(user: UserCreate):
    if await db.users.find_one({"email": user.email}):
        raise HTTPException(400, "Email déjà utilisé")
    user_id = str(uuid.uuid4())
    await db.users.insert_one({
        "user_id": user_id, "email": user.email,
        "password": hash_password(user.password),
        "tier": "free", "analyses_used": 0,
        "analyses_reset_at": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat()
    })
    return {"access_token": create_access_token({"sub": user_id}), "token_type": "bearer"}

@api_router.post("/login", response_model=Token)
async def login(user: UserCreate):
    db_user = await db.users.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(401, "Email ou mot de passe incorrect")
    return {"access_token": create_access_token({"sub": db_user["user_id"]}), "token_type": "bearer"}

@api_router.get("/me", response_model=UserResponse)
async def me(current_user: dict = Depends(get_current_user)):
    return current_user

MAX_DURATION_S = 60   # truncate audio to this before any processing
ANALYZE_TIMEOUT = 90  # hard timeout for the full pipeline (seconds)


def _truncate(y, sr, max_seconds=MAX_DURATION_S):
    """Truncate audio array to at most max_seconds."""
    max_samples = int(sr * max_seconds)
    if y.ndim == 1:
        return y[:max_samples] if len(y) > max_samples else y
    else:
        return y[:, :max_samples] if y.shape[1] > max_samples else y


async def _run_pipeline(tmp_path: str, filename: str, tier: str = "free"):
    """Full analysis pipeline — called by both /analyze and /analyze-stream."""
    t0 = time.time()
    logger.info(f"[analyze] loading: {filename}")
    y, sr = await asyncio.to_thread(librosa.load, tmp_path, sr=None, mono=False)
    logger.info(f"[analyze] load done in {time.time()-t0:.2f}s — shape {y.shape}")

    # Truncate to MAX_DURATION_S
    y = _truncate(y, sr)

    if y.ndim == 1:
        duration, channels, y_mono = len(y) / sr, 1, y
    else:
        duration, channels, y_mono = y.shape[1] / sr, y.shape[0], librosa.to_mono(y)
    logger.info(f"[analyze] duration after truncation: {duration:.1f}s")

    t1 = time.time()
    freq = await asyncio.to_thread(analyze_frequency_bands, y_mono, sr)
    logger.info(f"[analyze] freq bands {time.time()-t1:.2f}s")

    t2 = time.time()
    loudness = await asyncio.to_thread(analyze_loudness, y_mono, sr)
    logger.info(f"[analyze] loudness {time.time()-t2:.2f}s")

    t3 = time.time()
    stereo = await asyncio.to_thread(analyze_stereo, y, sr)
    logger.info(f"[analyze] stereo {time.time()-t3:.2f}s")

    t4 = time.time()
    spec3d = await asyncio.to_thread(generate_3d_spectrogram, y_mono, sr)
    logger.info(f"[analyze] spectrogram {time.time()-t4:.2f}s")

    score = calculate_score(freq, loudness, stereo)

    t5 = time.time()
    recs = await get_recommendations(freq, loudness, stereo, filename, tier=tier)
    logger.info(f"[analyze] claude {time.time()-t5:.2f}s — total {time.time()-t0:.2f}s")

    aid = str(uuid.uuid4())
    result = AnalysisResult(
        id=aid, filename=filename, duration=round(duration, 2),
        sample_rate=sr, channels=channels,
        frequency_analysis={k: v.model_dump() for k, v in freq.items()},
        loudness_analysis=loudness, stereo_analysis=stereo,
        recommendations=recs, overall_score=score, status="completed", spectrogram_3d=spec3d
    )
    doc = result.model_dump()
    doc['created_at'] = datetime.now(timezone.utc).isoformat()
    await db.audio_analyses.insert_one(doc)
    return result


@api_router.post("/analyze", response_model=AnalysisResult)
@limiter.limit("5/minute")
async def analyze_audio(request: Request, file: UploadFile = File(...),
                         current_user: dict = Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(400, "Nom de fichier manquant")
    fname = file.filename.lower()
    if not any(fname.endswith(f) for f in SUPPORTED_FORMATS):
        raise HTTPException(400, "Format non supporté. Acceptés: WAV, MP3, FLAC, AIFF, OGG")
    tier = current_user.get("tier", "free")
    if current_user.get("analyses_used", 0) >= TIER_LIMITS.get(tier, 2):
        raise HTTPException(403, f"Quota mensuel atteint ({TIER_LIMITS[tier]} analyses/{tier}). Passez au tier supérieur.")
    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, "Fichier trop volumineux (max 100 MB)")
        kind = filetype.guess(content)
        if not kind or kind.mime.split('/')[0] != 'audio':
            raise HTTPException(400, "Type de fichier invalide (format audio requis)")
        with tempfile.NamedTemporaryFile(suffix=Path(fname).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            result = await asyncio.wait_for(
                _run_pipeline(tmp_path, file.filename, tier=tier),
                timeout=ANALYZE_TIMEOUT
            )
            await db.audio_analyses.update_one(
                {"id": result.id}, {"$set": {"user_id": current_user["user_id"]}}
            )
            await db.users.update_one(
                {"user_id": current_user["user_id"]}, {"$inc": {"analyses_used": 1}}
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"[analyze] timeout after {ANALYZE_TIMEOUT}s for {file.filename}")
            raise HTTPException(504, f"Analyse trop longue (limite {ANALYZE_TIMEOUT}s). Essayez un fichier plus court.")
        finally:
            os.unlink(tmp_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, f"Erreur analyse: {str(e)}")


@api_router.post("/analyze-stream")
@limiter.limit("5/minute")
async def analyze_stream(request: Request, file: UploadFile = File(...),
                          current_user: dict = Depends(get_current_user)):
    if not file.filename:
        raise HTTPException(400, "Nom de fichier manquant")
    fname = file.filename.lower()
    if not any(fname.endswith(f) for f in SUPPORTED_FORMATS):
        raise HTTPException(400, "Format non supporté. Acceptés: WAV, MP3, FLAC, AIFF, OGG")
    tier = current_user.get("tier", "free")
    if current_user.get("analyses_used", 0) >= TIER_LIMITS.get(tier, 2):
        raise HTTPException(403, "Quota mensuel atteint. Passez au tier supérieur.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "Fichier trop volumineux (max 100 MB)")
    kind = filetype.guess(content)
    if not kind or kind.mime.split('/')[0] != 'audio':
        raise HTTPException(400, "Type de fichier invalide")
    filename = file.filename

    async def generate():
        def evt(step: str, progress: int, message: str = ""):
            return f"data: {json.dumps({'step': step, 'progress': progress, 'message': message})}\n\n"

        with tempfile.NamedTemporaryFile(suffix=Path(fname).suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            yield evt("loading", 10, "Chargement du fichier…")
            t0 = time.time()
            y, sr = await asyncio.to_thread(librosa.load, tmp_path, sr=None, mono=False)
            y = _truncate(y, sr)
            if y.ndim == 1:
                duration, channels, y_mono = len(y) / sr, 1, y
            else:
                duration, channels, y_mono = y.shape[1] / sr, y.shape[0], librosa.to_mono(y)

            yield evt("frequency", 30, "Analyse spectrale…")
            freq = await asyncio.to_thread(analyze_frequency_bands, y_mono, sr)

            yield evt("loudness", 50, "Analyse de la loudness…")
            loudness = await asyncio.to_thread(analyze_loudness, y_mono, sr)

            yield evt("stereo", 60, "Analyse stéréo…")
            stereo = await asyncio.to_thread(analyze_stereo, y, sr)

            yield evt("spectrogram", 70, "Génération du spectrogramme 3D…")
            spec3d = await asyncio.to_thread(generate_3d_spectrogram, y_mono, sr)

            score = calculate_score(freq, loudness, stereo)

            yield evt("recommendations", 85, "Recommandations IA…")
            recs = await get_recommendations(freq, loudness, stereo, filename, tier=tier)

            aid = str(uuid.uuid4())
            result = AnalysisResult(
                id=aid, filename=filename, duration=round(duration, 2),
                sample_rate=sr, channels=channels,
                frequency_analysis={k: v.model_dump() for k, v in freq.items()},
                loudness_analysis=loudness, stereo_analysis=stereo,
                recommendations=recs, overall_score=score, status="completed", spectrogram_3d=spec3d
            )
            result_dict = result.model_dump()
            result_dict['created_at'] = datetime.now(timezone.utc).isoformat()
            try:
                result_dict['user_id'] = current_user["user_id"]
                await db.audio_analyses.insert_one(dict(result_dict))
                await db.users.update_one(
                    {"user_id": current_user["user_id"]}, {"$inc": {"analyses_used": 1}}
                )
            except Exception as db_err:
                logger.error(f"[analyze-stream] DB insert failed (non-fatal): {db_err}")

            logger.info(f"[analyze-stream] done in {time.time()-t0:.2f}s")
            yield f"data: {json.dumps({'step': 'done', 'progress': 100, 'result': result_dict})}\n\n"
        except Exception as e:
            logger.error(f"[analyze-stream] error: {e}")
            yield f"data: {json.dumps({'step': 'error', 'message': str(e)})}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@api_router.get("/analyses", response_model=List[AnalysisResult])
async def get_analyses(current_user: dict = Depends(get_current_user)):
    try:
        return await db.audio_analyses.find(
            {"user_id": current_user["user_id"]}, {"_id": 0}
        ).sort([("created_at", -1)]).to_list(50)
    except Exception as e:
        logger.error(f"get_analyses error: {e}")
        raise HTTPException(500, "Erreur lors de la récupération des analyses")

@api_router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat_with_ai(request: Request, chat: ChatMessage,
                        current_user: dict = Depends(get_current_user)):
    try:
        system = """Tu es un ingénieur du son professionnel avec 20 ans d'expérience.
Tu réponds aux questions sur production musicale, mixage, mastering, acoustique, équipements.
Précis, technique mais accessible. En français. Hors sujet → redirige poliment vers l'audio."""
        if chat.context:
            system += f"\n\nContexte analyse:\n{chat.context}"

        messages = []
        if chat.history:
            for m in chat.history[-10:]:
                if m.get('role') in ('user','assistant') and m.get('content'):
                    messages.append({"role": m['role'], "content": m['content']})
        messages.append({"role": "user", "content": chat.message})

        def _call():
            return anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1024,
                system=system, messages=messages
            )
        msg = await asyncio.to_thread(_call)
        return ChatResponse(response=msg.content[0].text.strip())
    except anthropic.APIStatusError as e:
        logger.error(f"Chat Anthropic API error {e.status_code}: {e.message}")
        raise HTTPException(502, f"Erreur API IA ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError as e:
        logger.error(f"Chat Anthropic connection error: {e}")
        raise HTTPException(502, "Impossible de joindre l'API IA. Réessayez.")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(500, f"Erreur chat: {str(e)}")

@api_router.get("/export-pdf/{analysis_id}")
async def export_pdf(analysis_id: str, current_user: dict = Depends(get_current_user)):
    try:
        analysis = await db.audio_analyses.find_one(
            {"id": analysis_id, "user_id": current_user["user_id"]}, {"_id": 0}
        )
    except Exception as e:
        logger.error(f"export_pdf DB error: {e}")
        raise HTTPException(500, "Erreur base de données")
    if not analysis:
        raise HTTPException(404, "Analyse non trouvée")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm, leftMargin=20*mm, rightMargin=20*mm)
    styles = getSampleStyleSheet()
    S = lambda name, **kw: ParagraphStyle(name, parent=styles.get('Normal', styles['Normal']), **kw)
    title_s = S('T', fontSize=22, textColor=colors.HexColor('#6C63FF'), spaceAfter=12)
    h_s = S('H', fontSize=13, textColor=colors.HexColor('#6C63FF'), spaceBefore=14, spaceAfter=8)
    rec_s = S('R', fontSize=10, leftIndent=16, spaceAfter=7)
    foot_s = S('F', fontSize=8, textColor=colors.gray, alignment=TA_CENTER)

    el = []
    el.append(Paragraph("SoundMaster — Rapport d'Analyse Audio", title_s))
    el.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", S('sub', fontSize=9, textColor=colors.gray, spaceAfter=10)))

    # Infos
    el.append(Paragraph("Fichier", h_s))
    loudness = analysis['loudness_analysis']
    t = Table([["Fichier", analysis['filename']], ["Durée", f"{analysis['duration']:.1f}s"],
               ["Sample Rate", f"{analysis['sample_rate']} Hz"],
               ["Canaux", "Stéréo" if analysis['channels'] > 1 else "Mono"],
               ["Score", f"{analysis['overall_score']}/100"]], colWidths=[110, 280])
    t.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),10),('TEXTCOLOR',(0,0),(0,-1),colors.gray),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    el.append(t)

    # Loudness
    el.append(Paragraph("Analyse Loudness (ITU-R BS.1770-4)", h_s))
    t2 = Table([["LUFS Intégré", f"{loudness['lufs_integrated']:.1f} LUFS"],
                ["True Peak", f"{loudness['true_peak_db']:.1f} dBTP"],
                ["Peak", f"{loudness['peak_db']:.1f} dB"], ["RMS", f"{loudness['rms_db']:.1f} dB"],
                ["Headroom", f"{loudness['headroom_db']:.1f} dB"],
                ["Plage Dynamique", f"{loudness['dynamic_range_db']:.1f} dB"]], colWidths=[130,100])
    t2.setStyle(TableStyle([('FONTSIZE',(0,0),(-1,-1),10),('TEXTCOLOR',(0,0),(0,-1),colors.gray),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    el.append(t2)

    # Références
    el.append(Paragraph("Références de loudness", h_s))
    t3 = Table([["Plateforme","Cible"],["Spotify","-14 LUFS"],["Apple Music","-16 LUFS"],["YouTube","-14 LUFS"],["Club / DJ","-6 à -8 LUFS"]], colWidths=[130,100])
    t3.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#6C63FF')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                             ('FONTSIZE',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),5),
                             ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F5FF')])]))
    el.append(t3)

    # Spectral
    el.append(Paragraph("Analyse Spectrale", h_s))
    labels = {'sub_bass':'Sub Bass (20-60Hz)','bass':'Bass (60-250Hz)','low_mids':'Low Mids (250-500Hz)',
              'mids':'Mids (500-2kHz)','high_mids':'High Mids (2-4kHz)','highs':'Highs (4-8kHz)','air':'Air (8-20kHz)'}
    rows = [["Bande","Énergie","Pic"]] + [[labels.get(b,b), f"{d['energy_db']:.1f} dB", f"{d['peak_freq']:.0f} Hz"] for b,d in analysis['frequency_analysis'].items()]
    t4 = Table(rows, colWidths=[150,80,100])
    t4.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#6C63FF')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
                             ('FONTSIZE',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),5),
                             ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F5FF')])]))
    el.append(t4)

    # Recommandations
    el.append(Paragraph("Recommandations IA", h_s))
    for i, rec in enumerate(analysis['recommendations'], 1):
        el.append(Paragraph(f"<b>{i}.</b> {rec}", rec_s))

    el.append(Spacer(1, 20))
    el.append(Paragraph("SoundMaster · Propulsé par Claude (Anthropic)", foot_s))
    doc.build(el)
    buf.seek(0)
    fname_safe = analysis['filename'].rsplit('.',1)[0]
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=SoundMaster_{fname_safe}.pdf"})

app.include_router(api_router)
app.add_middleware(CORSMiddleware,
    allow_origins=[os.environ.get('FRONTEND_URL', 'http://localhost:5173')],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"])

@app.on_event("shutdown")
async def shutdown_db_client():
    client_db.close()
