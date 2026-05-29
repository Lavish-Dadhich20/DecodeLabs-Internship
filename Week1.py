import time
import random
class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    GRAY    = "\033[90m"
    WHITE   = "\033[97m"
    RED     = "\033[91m"

def colorize(text, color):
    return f"{color}{text}{Color.RESET}"

KNOWLEDGE_BASE = [
    {
        "keywords": ["hello", "hi", "hey", "hiya", "howdy", "sup", "greetings"],
        "reply": "Hey {name}! Great to have you here. What's on your mind?",
        "category": "GREETING"
    },
    {
        "keywords": ["who are you", "what are you", "your name", "introduce yourself"],
        "reply": "I'm ARIA — Automated Rule-based Intelligence Assistant, built at DecodeLabs!\nPowered by pure rule-based logic. No neural networks — just clean code.",
        "category": "IDENTITY"
    },
    {
        "keywords": ["how are you", "how do you feel", "are you okay"],
        "reply": "Running at 100% efficiency! Every rule matched, every response fired on time.",
        "category": "GREETING"
    },
    {
        "keywords": ["your name", "call you", "what do i call you"],
        "reply": "My name is ARIA. Nice to meet you, {name}!",
        "category": "IDENTITY"
    },
    {
        "keywords": ["what is ai", "define ai", "explain ai", "artificial intelligence"],
        "reply": "AI is the simulation of human intelligence by machines using logic, data & learning.\nI'm a great example — I use rule-based logic, the oldest form of AI!",
        "category": "KNOWLEDGE"
    },
    {
        "keywords": ["machine learning", "what is ml", "deep learning"],
        "reply": "Machine Learning is AI that learns from data without explicit programming.\nThat's Project 2 territory — first, we master the rules!",
        "category": "KNOWLEDGE"
    },
    {
        "keywords": ["decodelabs", "decode labs", "about decodelabs"],
        "reply": "DecodeLabs is an AI training platform in Greater Lucknow, India.\nIt runs structured internship programs to build real-world AI projects like this one!",
        "category": "KNOWLEDGE"
    },
    {
        "keywords": ["project 1", "this project", "chatbot project"],
        "reply": "Project 1 is the Rule-Based AI Chatbot — your foundation milestone.\nIt covers: control flow, IPO model, input sanitization, dictionary O(1) lookup & exit strategy.",
        "category": "KNOWLEDGE"
    },
    {
        "keywords": ["ipo model", "input process output", "how does this work"],
        "reply": "IPO Model:\n  INPUT  → Sanitize raw text (.lower().strip())\n  PROCESS → Match intent via dictionary (O(1))\n  OUTPUT → Generate & print the response\nThis is the architecture blueprint for ALL AI pipelines!",
        "category": "KNOWLEDGE"
    },
    {
        "keywords": ["o(1)", "big o", "time complexity", "why dictionary", "hash map"],
        "reply": "O(1) = constant time. A dictionary lookup is instant regardless of size.\nAn if-elif chain is O(n) — it checks each rule one by one.\nMore rules = slower. Dictionary = always fast. That's why we use it!",
        "category": "KNOWLEDGE"
    },
    {
        "keywords": ["joke", "funny", "make me laugh", "humor"],
        "reply": "Why do programmers prefer dark mode?\nBecause light attracts bugs! 🐛",
        "category": "FUN"
    },
    {
        "keywords": ["another joke", "more jokes", "funnier"],
        "reply": "A SQL query walks into a bar, walks up to two tables and asks:\n'Can I JOIN you?' 🍺",
        "category": "FUN"
    },
    {
        "keywords": ["help", "commands", "what can you do", "capabilities"],
        "reply": "I can help with:\n  • Questions about AI & ML\n  • DecodeLabs & Project 1 info\n  • The IPO model & O(1) efficiency\n  • Jokes & general chat\nTry: 'what is ai', 'ipo model', 'joke', or 'who are you'",
        "category": "HELP"
    },
    {
        "keywords": ["thank you", "thanks", "appreciate", "thx"],
        "reply": "Anytime, {name}! Every line of code you write today is your portfolio for tomorrow. 🚀",
        "category": "GREETING"
    },
    {
        "keywords": ["good morning", "morning"],
        "reply": "Good morning, {name}! Ready to build something intelligent today?",
        "category": "GREETING"
    },
    {
        "keywords": ["good night", "night", "sleep"],
        "reply": "Goodnight! Rest well — great engineers need great sleep. 🌙",
        "category": "GREETING"
    },
]

FALLBACK_REPLIES = [
    "Hmm, that's outside my current rule set. Try 'help' to see what I know!",
    "No rule matched for that input. Expand the KNOWLEDGE_BASE to handle it!",
    "I don't have a response for that yet — that's exactly why Project 2 (ML) exists!",
    "Intent not recognized. This is what a fallback handler is for. Try 'help'!",
]

EXIT_KEYWORDS = {"exit", "quit", "bye", "goodbye", "stop", "see you", "later"}


def sanitize(raw: str) -> str:
    """Normalize input: lowercase + strip whitespace."""
    return raw.strip().lower()


def match_intent(clean_input: str, context: dict) -> dict:
    """
    Scans each rule's keyword list.
    Fires if the user input CONTAINS any keyword (partial match).
    Falls back to rotating fallback messages.
    Returns: {reply, category}
    """
    for rule in KNOWLEDGE_BASE:
        for keyword in rule["keywords"]:
            if keyword in clean_input:
                reply = rule["reply"].format(name=context.get("name", "friend"))
                return {"reply": reply, "category": rule["category"]}

    fb = FALLBACK_REPLIES[context["fallback_idx"] % len(FALLBACK_REPLIES)]
    context["fallback_idx"] += 1
    return {"reply": fb, "category": "FALLBACK"}


CATEGORY_COLORS = {
    "GREETING" : Color.GREEN,
    "IDENTITY" : Color.CYAN,
    "KNOWLEDGE": Color.CYAN,
    "FUN"      : Color.YELLOW,
    "HELP"     : Color.GREEN,
    "FALLBACK" : Color.GRAY,
    "EXIT"     : Color.RED,
}

def display_response(result: dict) -> None:
    cat_color = CATEGORY_COLORS.get(result["category"], Color.WHITE)
    cat_label = colorize(f"[{result['category']}]", cat_color)
    time.sleep(0.4)  
    print(f"\n  {colorize('ARIA', Color.CYAN + Color.BOLD)} {cat_label}")
    for line in result["reply"].split("\n"):
        print(f"  {line}")
    print()


def check_for_name(raw_input: str, context: dict) -> None:
    """
    Detects if user introduces themselves.
    e.g. 'my name is ravi' → context['name'] = 'Ravi'
    """
    triggers = ["my name is", "i am", "i'm", "call me"]
    lowered = raw_input.lower()
    for t in triggers:
        if t in lowered:
            parts = lowered.split(t)
            if len(parts) > 1:
                name = parts[1].strip().split()[0].capitalize()
                if len(name) > 1:
                    context["name"] = name
                    print(f"\n  {colorize('ARIA', Color.CYAN + Color.BOLD)} {colorize('[MEMORY]', Color.YELLOW)}")
                    print(f"  Nice to meet you, {colorize(name, Color.GREEN)}! I'll remember your name.\n")


def run_chatbot() -> None:
    context = {
        "name": "friend",
        "fallback_idx": 0,
        "turn_count": 0,
    }

    print("\n" + "=" * 58)
    print(colorize("  ARIA — Rule-Based AI Chatbot  |  DecodeLabs 2026", Color.CYAN + Color.BOLD))
    print(colorize("  Architecture: IPO Model  |  Dictionary O(1) Lookup", Color.GRAY))
    print("  " + colorize("Type 'help' for commands", Color.GREEN) + "  |  " + colorize("'exit' to quit", Color.RED))
    print("=" * 58 + "\n")

    display_response({"reply": "Hello! I'm ARIA, your AI assistant at DecodeLabs.\nWhat would you like to know today?", "category": "GREETING"})

    while True:

        try:
            raw = input(colorize("  You: ", Color.WHITE + Color.BOLD))
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        context["turn_count"] += 1
        clean = sanitize(raw)

        if not clean:
            print(f"  {colorize('ARIA', Color.CYAN)}: Please type something. Try 'help'!\n")
            continue

        if any(ex in clean for ex in EXIT_KEYWORDS):
            display_response({"reply": f"Goodbye, {context['name']}! Keep coding and stay curious 👋\n  Total turns this session: {context['turn_count']}", "category": "EXIT"})
            break

        check_for_name(clean, context)

        result = match_intent(clean, context)

        display_response(result)


if __name__ == "__main__":
    run_chatbot()
