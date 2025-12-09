# Speaker Diarization Service

A Python microservice for speaker diarization and transcription using pyannote.audio and Whisper.

## Features

- **Speaker Diarization**: Identifies who spoke when using pyannote.audio 3.1
- **Transcription**: High-quality transcription using faster-whisper
- **Voice Embeddings**: Extracts speaker embeddings for future voice recognition
- **Async Processing**: Queue-based processing for long recordings
- **Webhook Callbacks**: Notifies Next.js app when processing completes

## Prerequisites

1. **HuggingFace Account**: Required for pyannote models
   - Create account at https://huggingface.co
   - Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept terms at https://huggingface.co/pyannote/segmentation-3.0
   - Create access token at https://huggingface.co/settings/tokens

2. **Python 3.10+**

3. **FFmpeg**: For audio processing
   ```bash
   # macOS
   brew install ffmpeg

   # Ubuntu/Debian
   apt-get install ffmpeg
   ```

## Local Development

1. **Create virtual environment**:
   ```bash
   cd services/diarization
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your HUGGINGFACE_TOKEN
   ```

4. **Run the service**:
   ```bash
   python main.py
   ```

   The service will start at http://localhost:8000

5. **Test the health endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

## API Endpoints

### POST /diarize
Queue an audio file for async diarization.

```bash
curl -X POST http://localhost:8000/diarize \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "abc123",
    "audio_url": "https://example.com/audio.webm",
    "callback_url": "http://localhost:3000/api/webhooks/diarization"
  }'
```

### POST /diarize/sync
Synchronously diarize an audio file (for short recordings < 5 min).

### GET /jobs/{job_id}
Check status of a diarization job.

### GET /health
Health check endpoint.

## Deployment to Railway

1. **Create new Railway project** or add to existing

2. **Set environment variables**:
   - `HUGGINGFACE_TOKEN`: Your HuggingFace access token
   - `WEBHOOK_URL`: Your Next.js app webhook URL
   - `WEBHOOK_SECRET`: Secret for webhook authentication
   - `DEVICE`: "cpu" (or "cuda" if using GPU instance)
   - `WHISPER_MODEL`: "base" (or larger for better accuracy)

3. **Deploy**:
   ```bash
   railway up
   ```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_TOKEN` | HuggingFace access token | Required |
| `WEBHOOK_URL` | Callback URL for results | http://localhost:3000/api/webhooks/diarization |
| `WEBHOOK_SECRET` | Secret for webhook auth | "" |
| `REDIS_URL` | Redis URL for job queue | None (uses in-memory) |
| `DEVICE` | "cpu" or "cuda" | "cpu" |
| `WHISPER_MODEL` | tiny/base/small/medium/large-v3 | "base" |
| `MIN_SPEAKERS` | Minimum speakers to detect | None |
| `MAX_SPEAKERS` | Maximum speakers to detect | None |
| `HOST` | Server host | "0.0.0.0" |
| `PORT` | Server port | 8000 |

## Processing Pipeline

1. **Audio Download**: Fetches audio from provided URL
2. **Diarization**: pyannote identifies speaker segments
3. **Transcription**: Whisper transcribes the audio
4. **Alignment**: Matches transcript text to speaker segments
5. **Embeddings**: Extracts voice embeddings for each speaker
6. **Webhook**: Sends results to callback URL

## Response Format

```json
{
  "recording_id": "abc123",
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "speaker": "SPEAKER_00",
      "text": "Hello, how are you?"
    },
    {
      "start": 3.5,
      "end": 7.2,
      "speaker": "SPEAKER_01",
      "text": "I'm doing great, thanks for asking."
    }
  ],
  "speaker_count": 2,
  "embeddings": {
    "SPEAKER_00": [0.123, -0.456, ...],
    "SPEAKER_01": [0.789, 0.012, ...]
  },
  "full_transcript": "Hello, how are you? I'm doing great, thanks for asking."
}
```

## Performance Notes

- **CPU Processing**: ~2-3x real-time (1 min audio = 2-3 min processing)
- **GPU Processing**: ~0.3-0.5x real-time (1 min audio = 20-30s processing)
- **Memory**: Requires ~4GB RAM for base model, more for larger models
- **First Request**: Slower due to model loading (~30-60s)

## Troubleshooting

### "No module named 'pyannote'"
Ensure you've installed requirements in your virtual environment.

### "Could not find speaker-diarization-3.1"
Accept the model terms on HuggingFace and verify your token.

### Out of Memory
- Reduce `WHISPER_MODEL` size (use "tiny" or "base")
- Process shorter audio files
- Use GPU instance on Railway
