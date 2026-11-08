"""Flask backend: POST /api/chat, DELETE /api/conversations/<id>, GET /api/health, and the web UI."""
import logging

from flask import Flask, jsonify, request, send_from_directory

import config
from chat_core import ChatService
from providers import LLMError, get_provider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chatbot")


def create_app(service: ChatService | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    # Build the service lazily-but-early so a missing API key is reported at startup.
    app.config["SERVICE"] = service or ChatService(get_provider())

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        svc: ChatService = app.config["SERVICE"]
        return jsonify(status="ok", provider=svc.provider.name, strategy=svc.strategy, bot=config.BOT_NAME)

    @app.post("/api/chat")
    def chat():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Send JSON like {\"message\": \"hi\"}.", kind="bad_request"), 400
        try:
            result = app.config["SERVICE"].chat(data.get("conversation_id"), data.get("message"))
            return jsonify(result)
        except ValueError as exc:
            return jsonify(error=str(exc), kind="bad_request"), 400
        except LLMError as exc:
            log.warning("LLM error (%s): %s", exc.kind, exc)
            return jsonify(error=exc.user_message, kind=exc.kind), exc.status
        except Exception:  # noqa: BLE001 - never leak a stack trace to the browser
            log.exception("Unexpected error in /api/chat")
            return jsonify(error="Something went wrong on the server. Please try again.", kind="server_error"), 500

    @app.delete("/api/conversations/<conversation_id>")
    def reset(conversation_id):
        app.config["SERVICE"].reset(conversation_id)
        return jsonify(status="reset")

    return app


if __name__ == "__main__":
    try:
        application = create_app()
    except RuntimeError as exc:  # e.g. missing API key
        raise SystemExit(f"\nSetup problem: {exc}\n")
    print(f"\n  {config.BOT_NAME} is running -> http://127.0.0.1:5000\n")
    application.run(host="127.0.0.1", port=5000, debug=False)
