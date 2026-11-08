"""Conversation storage + context-window management (framework independent)."""
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import config
from providers import BaseProvider, LLMError

SUMMARIZER_SYSTEM = (
    "You compress conversations into a running summary for an AI assistant's memory. "
    "Keep every durable fact: the user's name, goals, background, preferences, decisions, "
    "open questions and anything the assistant promised. Drop pleasantries. "
    "Write plain text, at most 180 words, no preamble."
)


@dataclass
class Conversation:
    id: str
    messages: list = field(default_factory=list)  # [{"role": "user"|"assistant", "content": str}]
    summary: str = ""  # running summary of messages that were trimmed away
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class ConversationStore:
    """Thread-safe in-memory store with an LRU cap so it can't grow forever."""

    def __init__(self, max_conversations: int = config.MAX_CONVERSATIONS):
        self._items: "OrderedDict[str, Conversation]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_conversations

    def get_or_create(self, conversation_id: str | None) -> Conversation:
        with self._lock:
            if conversation_id and conversation_id in self._items:
                self._items.move_to_end(conversation_id)
                return self._items[conversation_id]
            convo = Conversation(id=conversation_id or uuid.uuid4().hex)
            self._items[convo.id] = convo
            while len(self._items) > self._max:
                self._items.popitem(last=False)
            return convo

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._items.pop(conversation_id, None) is not None


def _transcript(messages: list[dict]) -> str:
    return "\n".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages)


class ChatService:
    def __init__(
        self,
        provider: BaseProvider,
        store: ConversationStore | None = None,
        strategy: str = config.CONTEXT_STRATEGY,
        max_messages: int = config.MAX_HISTORY_MESSAGES,
        keep_recent: int = config.KEEP_RECENT_MESSAGES,
        system_prompt: str = config.SYSTEM_PROMPT,
    ):
        if keep_recent >= max_messages:
            raise ValueError("keep_recent must be smaller than max_messages")
        self.provider = provider
        self.store = store or ConversationStore()
        self.strategy = strategy
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------ context
    def _compact(self, convo: Conversation) -> bool:
        """Shrink history if it is too long. Returns True if anything was trimmed."""
        if len(convo.messages) <= self.max_messages:
            return False

        older = convo.messages[: -self.keep_recent]
        recent = convo.messages[-self.keep_recent :]

        if self.strategy == "summary":
            try:
                prompt = (
                    f"Existing summary:\n{convo.summary or '(none)'}\n\n"
                    f"New messages to fold in:\n{_transcript(older)}\n\n"
                    "Write the updated summary."
                )
                convo.summary = self.provider.generate(
                    SUMMARIZER_SYSTEM, [{"role": "user", "content": prompt}]
                )
            except LLMError:
                # Summarising failed (rate limit etc.): fall back to plain truncation
                # so the main request can still go through.
                pass

        # Chat APIs expect the history to start with a user turn.
        while recent and recent[0]["role"] != "user":
            recent.pop(0)
        convo.messages = recent
        return True

    def _system_for(self, convo: Conversation) -> str:
        if not convo.summary:
            return self.system_prompt
        return (
            f"{self.system_prompt}\n\n"
            "Summary of the earlier part of this conversation (older messages were trimmed; "
            f"treat it as things you already know):\n{convo.summary}"
        )

    # --------------------------------------------------------------------- chat
    def chat(self, conversation_id: str | None, user_message: str) -> dict:
        user_message = (user_message or "").strip()
        if not user_message:
            raise ValueError("Message is empty.")
        if len(user_message) > config.MAX_USER_MESSAGE_CHARS:
            raise ValueError(f"Message is too long (max {config.MAX_USER_MESSAGE_CHARS} characters).")

        convo = self.store.get_or_create(conversation_id)
        with convo.lock:  # one in-flight request per conversation keeps history consistent
            compacted = self._compact(convo)
            convo.messages.append({"role": "user", "content": user_message})
            try:
                reply = self.provider.generate(self._system_for(convo), convo.messages)
            except LLMError:
                convo.messages.pop()  # roll back so a retry doesn't duplicate the message
                raise
            convo.messages.append({"role": "assistant", "content": reply})
            return {
                "conversation_id": convo.id,
                "reply": reply,
                "context": {
                    "strategy": self.strategy,
                    "messages_in_context": len(convo.messages),
                    "has_summary": bool(convo.summary),
                    "compacted_this_turn": compacted,
                },
            }

    def reset(self, conversation_id: str) -> bool:
        return self.store.delete(conversation_id)
