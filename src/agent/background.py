"""Run the agent in a background thread so that a web page can show its progress and answer its questions."""

import threading
import traceback

from src.agent.graph import run_agent
from src.agent.io import AutoAsker
from src.models import TabPFNSetupError


class WebAsker:
    """A confirmation point that waits until the page answers (or the run is cancelled).

    ``pending`` holds the open question (``id``, ``kind``, ``title``, ``body``) or None.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._answer = None
        self._count = 0
        self.pending = None
        self.cancelled = False

    def ask(self, kind, title, body, default="y"):
        with self._cond:
            if self.cancelled:
                return "n"
            self._count += 1
            self._answer = None
            self.pending = {"id": self._count, "kind": kind, "title": title, "body": body}
            while self._answer is None and not self.cancelled:
                self._cond.wait(timeout=1.0)
            answer = "n" if self._answer is None else self._answer
            self.pending = None
            return answer

    def answer(self, text):
        """Answer the open question. It is cleared at once, so a page never shows a question that was answered."""
        with self._cond:
            if self.pending is None:
                return
            self._answer = text.strip() or "y"
            self.pending = None
            self._cond.notify_all()

    def cancel(self):
        """Answer 'n' to the open question and to every later one: the run stops at the next confirmation point."""
        with self._cond:
            self.cancelled = True
            self._cond.notify_all()


class AgentJob:
    """One agent run in a daemon thread.

    Parameters
    ----------
    trial, hist : pandas.DataFrame
    auto : bool
        Confirm every step automatically (``--yes``) instead of waiting for the page.
    **kwargs
        Passed to :func:`src.agent.run_agent` (``nct``, ``eligibility_text``, ``backend``, ``llm`` ...).

    Attributes
    ----------
    log : list of str
        Progress lines. ``result`` (the result dict) or ``error`` (text) is set when the run ends.
    """

    def __init__(self, trial, hist, auto=False, **kwargs):
        self.asker = AutoAsker() if auto else WebAsker()
        self.log, self.result, self.error = [], None, None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, args=(trial, hist, kwargs), daemon=True)
        self._thread.start()

    def _say(self, message):
        with self._lock:
            self.log.append(str(message))

    def _run(self, trial, hist, kwargs):
        try:
            self.result = run_agent(trial, hist, asker=self.asker, say=self._say, **kwargs)
        except TabPFNSetupError as e:
            self.error = f"TabPFN is not set up: {e}"
        except Exception as e:  # noqa: BLE001 - shown on the page instead of dying silently in the thread
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"

    @property
    def running(self):
        return self._thread.is_alive()

    @property
    def pending(self):
        return getattr(self.asker, "pending", None)

    def lines(self, n=60):
        with self._lock:
            return list(self.log[-n:])

    def answer(self, text):
        self.asker.answer(text)

    def cancel(self):
        if isinstance(self.asker, WebAsker):
            self.asker.cancel()

    def join(self, timeout=None):
        self._thread.join(timeout)
