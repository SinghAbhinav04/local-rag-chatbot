"""
Text-to-speech via macOS `say` command.

A daemon thread continuously drains a queue of sentences, so TTS
never blocks the main chat loop.  Toggle `voice_enabled` from other
modules (e.g. `speech.voice_enabled = False`) to mute/unmute.
"""

import queue
import subprocess
import threading


# ──────────────────────────────────────────
# State (module-level so other modules can toggle)
# ──────────────────────────────────────────

speech_queue  = queue.Queue()
voice_enabled = True


# ──────────────────────────────────────────
# Worker thread (daemon — dies with the process)
# ──────────────────────────────────────────

def _speech_worker():
    """Background loop: pull sentences from the queue and speak them."""
    while True:
        text = speech_queue.get()
        if text is None:
            break
        subprocess.run(["say", "-v", "Samantha", text])
        speech_queue.task_done()


# Start the daemon thread the moment this module is imported.
# Because it's a daemon, it won't prevent the process from exiting.
threading.Thread(target=_speech_worker, daemon=True).start()


# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────

def speak(text: str):
    """Enqueue a sentence for TTS (no-op when voice is disabled)."""
    if voice_enabled:
        speech_queue.put(text)


def stop_speaking():
    """Drain any queued TTS sentences and kill the current 'say' process."""
    while not speech_queue.empty():
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except Exception:
            break
    # Kill any currently running 'say' process
    subprocess.run(["pkill", "-x", "say"], capture_output=True)
