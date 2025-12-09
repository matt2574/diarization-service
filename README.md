# Speaker Diarization Service

A lightweight Python microservice that proxies requests to pyannoteAI's cloud API for speaker diarization.

## Features

- **Speaker Diarization**: Identifies who spoke when using pyannoteAI
- **Speaker Identification**: Match voices against known voiceprints
- **Voiceprint Creation**: Create voice profiles for speaker recognition
- **Webhook Callbacks**: Notifies Next.js app when processing completes

## Why pyannoteAI?

- **Fast**: 1 hour of audio processed in ~14 seconds
- **Accurate**: State-of-the-art precision-2 model
- **Voiceprints**: Built-in speaker recognition across recordings
- **Simple**: No GPU, no model management, just API calls

## Prerequisites

1. **pyannoteAI Account**: Get an API key at https://dashboard.pyannote.ai
   - Free trial: 150 hours of processing

2. **Python 3.10+**

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
   # Edit .env and add your PYANNOTE_API_KEY
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
Submit an audio file for speaker diarization.

```bash
curl -X POST http://localhost:8000/diarize \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "abc123",
    "audio_url": "https://example.com/audio.webm",
    "callback_url": "http://localhost:3000/api/webhooks/diarization"
  }'
```

### POST /identify
Submit an audio file for diarization with speaker identification.

```bash
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{
    "recording_id": "abc123",
    "audio_url": "https://example.com/audio.webm",
    "voiceprints": [
      {"label": "John", "voiceprint": "base64-encoded-voiceprint"}
    ],
    "callback_url": "http://localhost:3000/api/webhooks/diarization"
  }'
```

### POST /voiceprint
Create a voiceprint from an audio sample (max 30 seconds).

### GET /jobs/{job_id}
Check status of a diarization job.

### GET /health
Health check endpoint.

## Deployment to Railway

1. **Create new Railway project** or add to existing

2. **Set environment variables**:
   - `PYANNOTE_API_KEY`: Your pyannoteAI API key
   - `WEBHOOK_URL`: Your Next.js app webhook URL
   - `WEBHOOK_SECRET`: Secret for webhook authentication
   - `PYANNOTE_MODEL`: `precision-2` (default), `precision-1`, or `community-1`

3. **Deploy**:
   ```bash
   railway up
   ```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PYANNOTE_API_KEY` | pyannoteAI API key | Required |
| `WEBHOOK_URL` | Callback URL for results | http://localhost:3000/api/webhooks/diarization |
| `WEBHOOK_SECRET` | Secret for webhook auth | "" |
| `PYANNOTE_MODEL` | Model to use | "precision-2" |
| `HOST` | Server host | "0.0.0.0" |
| `PORT` | Server port | 8000 |

## Models

| Model | Speed | Accuracy | Cost |
|-------|-------|----------|------|
| `precision-2` | Fastest | Best | Premium |
| `precision-1` | Fast | Great | Premium |
| `community-1` | Fast | Good | Cheapest |

## Response Format (from webhook)

pyannoteAI sends results directly to your webhook:

```json
{
  "jobId": "abc123",
  "status": "succeeded",
  "output": {
    "diarization": [
      {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.5},
      {"speaker": "SPEAKER_01", "start": 3.5, "end": 7.2}
    ]
  }
}
```

When using `/identify` with voiceprints, speakers are labeled with your provided names instead of `SPEAKER_XX`.

## Pricing

- **Free trial**: 150 hours
- **Developer plan**: €19/month for 125 hours
- See https://pyannote.ai/pricing for current rates

## Architecture

```
Next.js App                    This Service              pyannoteAI
     │                              │                        │
     │  POST /diarize               │                        │
     │  {recording_id, audio_url}   │                        │
     ├─────────────────────────────►│                        │
     │                              │  POST /v1/diarize      │
     │                              │  {url, webhook}        │
     │                              ├───────────────────────►│
     │                              │                        │
     │                              │  {jobId, status}       │
     │                              │◄───────────────────────┤
     │  {job_id, status}            │                        │
     │◄─────────────────────────────┤                        │
     │                              │                        │
     │                              │        ... processing ...
     │                              │                        │
     │  POST /webhooks/diarization  │                        │
     │  {jobId, output}             │                        │
     │◄─────────────────────────────┼────────────────────────┤
     │                              │                        │
```
