"""Job queue for async processing of audio files."""

import uuid
import json
import threading
from datetime import datetime
from typing import Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from config import get_settings


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents a diarization job."""

    id: str
    recording_id: str
    audio_url: str
    callback_url: str
    status: JobStatus = JobStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


class InMemoryQueue:
    """Simple in-memory job queue for development/single-instance deployments."""

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add_job(
        self,
        recording_id: str,
        audio_url: str,
        callback_url: str,
    ) -> Job:
        """Add a new job to the queue."""
        job = Job(
            id=str(uuid.uuid4()),
            recording_id=recording_id,
            audio_url=audio_url,
            callback_url=callback_url,
        )

        with self._lock:
            self.jobs[job.id] = job

        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self.jobs.get(job_id)

    def get_next_pending(self) -> Job | None:
        """Get the next pending job."""
        with self._lock:
            for job in self.jobs.values():
                if job.status == JobStatus.PENDING:
                    return job
        return None

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        """Update a job's status and result."""
        job = self.jobs.get(job_id)
        if not job:
            return

        with self._lock:
            if status is not None:
                job.status = status
                if status == JobStatus.PROCESSING:
                    job.started_at = datetime.utcnow().isoformat()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = datetime.utcnow().isoformat()

            if result is not None:
                job.result = result

            if error is not None:
                job.error = error


class RedisQueue:
    """Redis-backed job queue for production/multi-instance deployments."""

    def __init__(self, redis_url: str):
        import redis

        self.redis = redis.from_url(redis_url)
        self.queue_key = "diarization:jobs"
        self.job_prefix = "diarization:job:"

    def add_job(
        self,
        recording_id: str,
        audio_url: str,
        callback_url: str,
    ) -> Job:
        """Add a new job to the queue."""
        job = Job(
            id=str(uuid.uuid4()),
            recording_id=recording_id,
            audio_url=audio_url,
            callback_url=callback_url,
        )

        # Store job data
        self.redis.set(f"{self.job_prefix}{job.id}", json.dumps(job.to_dict()))

        # Add to queue
        self.redis.rpush(self.queue_key, job.id)

        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        data = self.redis.get(f"{self.job_prefix}{job_id}")
        if not data:
            return None

        job_dict = json.loads(data)
        job_dict["status"] = JobStatus(job_dict["status"])
        return Job(**job_dict)

    def get_next_pending(self) -> Job | None:
        """Get the next pending job from the queue."""
        job_id = self.redis.lpop(self.queue_key)
        if not job_id:
            return None

        return self.get_job(job_id.decode())

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ):
        """Update a job's status and result."""
        job = self.get_job(job_id)
        if not job:
            return

        if status is not None:
            job.status = status
            if status == JobStatus.PROCESSING:
                job.started_at = datetime.utcnow().isoformat()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.utcnow().isoformat()

        if result is not None:
            job.result = result

        if error is not None:
            job.error = error

        self.redis.set(f"{self.job_prefix}{job.id}", json.dumps(job.to_dict()))


# Global queue instance
_queue: InMemoryQueue | RedisQueue | None = None


def get_queue() -> InMemoryQueue | RedisQueue:
    """Get the global queue instance."""
    global _queue
    if _queue is None:
        settings = get_settings()
        if settings.redis_url:
            _queue = RedisQueue(settings.redis_url)
        else:
            _queue = InMemoryQueue()
    return _queue
