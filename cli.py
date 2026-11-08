"""Terminal version of the chatbot.  Run:  python cli.py"""
import config
from chat_core import ChatService
from providers import LLMError, get_provider

HELP = "Commands: /reset  start a new conversation | /history  show messages in context | /quit  exit"


def main() -> None:
    try:
        service = ChatService(get_provider())
    except RuntimeError as exc:
        raise SystemExit(f"\nSetup problem: {exc}\n")

    print(f"\n{config.BOT_NAME} ({service.provider.name}, context strategy: {service.strategy})")
    print(HELP + "\n")

    conversation_id = None
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return
        if not text:
            continue
        if text == "/quit":
            print("bye!")
            return
        if text == "/reset":
            if conversation_id:
                service.reset(conversation_id)
            conversation_id = None
            print("(new conversation)\n")
            continue
        if text == "/history":
            convo = service.store.get_or_create(conversation_id) if conversation_id else None
            if convo is None or not (convo.messages or convo.summary):
                print("(empty)\n")
                continue
            if convo.summary:
                print(f"[summary] {convo.summary}")
            for m in convo.messages:
                print(f"[{m['role']}] {m['content']}")
            print()
            continue

        try:
            result = service.chat(conversation_id, text)
        except ValueError as exc:
            print(f"! {exc}\n")
            continue
        except LLMError as exc:
            print(f"! {exc.user_message}\n")
            continue
        conversation_id = result["conversation_id"]
        print(f"\n{config.BOT_NAME} > {result['reply']}\n")


if __name__ == "__main__":
    main()
