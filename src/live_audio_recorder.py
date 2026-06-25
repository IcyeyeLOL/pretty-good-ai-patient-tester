from __future__ import annotations

import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

from src.recorder import get_run_directory, update_call_meta


SAMPLE_RATE = 8000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1


class _AlignedWaveTrack:
    """Writes mono PCM audio aligned to wall-clock call time."""

    def __init__(self, path: Path, started_at: float):
        self.path = path
        self.started_at = started_at
        self.samples_written = 0
        self.audio_bytes_written = 0
        self._lock = threading.Lock()
        self._wave = wave.open(str(path), "wb")
        self._wave.setnchannels(CHANNELS)
        self._wave.setsampwidth(SAMPLE_WIDTH_BYTES)
        self._wave.setframerate(SAMPLE_RATE)

    def write(self, audio: bytes, sample_rate: int, num_channels: int) -> None:
        if not audio or sample_rate != SAMPLE_RATE or num_channels != CHANNELS:
            return
        with self._lock:
            elapsed_samples = max(0, int((time.monotonic() - self.started_at) * SAMPLE_RATE))
            silence_samples = elapsed_samples - self.samples_written
            if silence_samples > 0:
                self._wave.writeframes(b"\x00" * silence_samples * SAMPLE_WIDTH_BYTES)
                self.samples_written += silence_samples

            self._wave.writeframes(audio)
            samples = len(audio) // SAMPLE_WIDTH_BYTES
            self.samples_written += samples
            self.audio_bytes_written += len(audio)

    def close(self) -> None:
        with self._lock:
            self._wave.close()

    @property
    def has_audio(self) -> bool:
        return self.audio_bytes_written > 0


class LiveAudioRecorder:
    """Captures inbound/outbound live PCM and exports a stereo MP3."""

    def __init__(self, scenario_id: int, call_sid: str, started_at: float | None = None):
        self.scenario_id = scenario_id
        self.call_sid = call_sid
        self.started_at = started_at if started_at is not None else time.monotonic()
        self.run_dir = Path(get_run_directory(scenario_id, call_sid))
        self.agent_wav = self.run_dir / "live_agent.wav"
        self.patient_wav = self.run_dir / "live_patient.wav"
        self.recording_mp3 = self.run_dir / "recording.mp3"
        self._closed = False
        self._agent = _AlignedWaveTrack(self.agent_wav, self.started_at)
        self._patient = _AlignedWaveTrack(self.patient_wav, self.started_at)

    def write_agent(self, audio: bytes, sample_rate: int, num_channels: int) -> None:
        if not self._closed:
            self._agent.write(audio, sample_rate, num_channels)

    def write_patient(self, audio: bytes, sample_rate: int, num_channels: int) -> None:
        if not self._closed:
            self._patient.write(audio, sample_rate, num_channels)

    def close(self) -> str | None:
        if self._closed:
            return str(self.recording_mp3) if self.recording_mp3.exists() else None
        self._closed = True
        self._agent.close()
        self._patient.close()

        mp3_path = self._export_mp3()
        extra = {
            "recording_source": "live_websocket",
            "live_agent_wav": str(self.agent_wav),
            "live_patient_wav": str(self.patient_wav),
        }
        if mp3_path:
            extra["recording_path"] = mp3_path
        update_call_meta(self.scenario_id, self.call_sid, extra=extra)
        return mp3_path

    def _export_mp3(self) -> str | None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._write_error("ffmpeg was not found; kept live_agent.wav and live_patient.wav only.")
            return None

        if self._agent.has_audio and self._patient.has_audio:
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.agent_wav),
                "-i",
                str(self.patient_wav),
                "-filter_complex",
                "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
                "-map",
                "[a]",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "96k",
                str(self.recording_mp3),
            ]
        elif self._agent.has_audio:
            cmd = self._mono_mp3_cmd(ffmpeg, self.agent_wav)
        elif self._patient.has_audio:
            cmd = self._mono_mp3_cmd(ffmpeg, self.patient_wav)
        else:
            self._write_error("No live audio frames were captured.")
            return None

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            return str(self.recording_mp3)
        except Exception as exc:
            self._write_error(str(exc))
            return None

    def _mono_mp3_cmd(self, ffmpeg: str, wav_path: Path) -> list[str]:
        return [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(self.recording_mp3),
        ]

    def _write_error(self, message: str) -> None:
        (self.run_dir / "live_recording_error.txt").write_text(message, encoding="utf-8")
