"""
Utilities Module
-----------------
Contains all utility functions for the local voice AI agent.
"""

# ============================================================================
# FUNCTIONS
# ============================================================================

#Extract everything after "Transcript:" from the conversation
def extract_transcript(text: str) -> str:
    marker = "Transcript:"
    idx = text.find(marker)
    if idx == -1:
        return ""

    line_end = text.find("\n", idx)
    if line_end == -1:
        return ""

    return text[line_end + 1:]

import re

def extract_last_replies(text: str, n: int = 4) -> list[str]:
    # Matches lines that start with optional whitespace + optional quote + (User:|AI:)
    # Captures the whole line only (no DOTALL needed)
    pattern = r'^[ \t"]*(User:|AI:)[ \t"]*(.*)$'
    matches = re.findall(pattern, text, flags=re.MULTILINE)

    replies = []
    for speaker, content in matches:
        # Keep content as-is (except trimming trailing spaces)
        content = content.rstrip()
        replies.append(f"{speaker} {content}" if content else speaker)

    return replies[-n:]
def back_and_forth(transcript: str, n:int=4) -> str:
    last = extract_last_replies(transcript, n)
    if len(last) < n:
        return False

    prev = None
    for r in last:
        speaker = r.split(":", 1)[0]
        if speaker not in ("AI", "User"):
            return False
        if prev is not None and speaker == prev:
            return False
        prev = speaker

    return True

example_conversation = """
Transcript:

User: Hello, how are you?
AI: I'm doing great, thank you! How about you?
User: I'm good, thanks. What's new?
AI: Not much, just working on a new project.
User: That sounds interesting. What is it?
AI: It's a new chatbot that I'm building.
User: Cool! How does it work?
AI: It uses a combination of natural language processing and machine learning to understand user intent and respond appropriately.
User: That sounds like a lot of work.
AI: Yeah, it is. But it's also a lot of fun.
"""

# ============================================================================
# AI-DIRECTED SPEECH DETECTION
# ============================================================================

# Patterns that indicate speech is directed at the AI
AI_TRIGGER_PATTERNS = [
    r'\?$',                                # Questions
    r'^(hey|hi|hello)\s*(ai|assistant)?',  # Greetings
    r'what do you think',
    r'tell me',
    r'can you',
    r'do you know',
]

def is_directed_at_ai(transcript: str) -> bool:
    """
    Detect if speech is directed at the AI.

    Args:
        transcript: The transcribed speech to analyze

    Returns:
        True if speech appears directed at AI, False otherwise
    """
    transcript_lower = transcript.lower().strip()
    if transcript_lower.endswith('?'):
        return True
    for pattern in AI_TRIGGER_PATTERNS:
        if re.search(pattern, transcript_lower, re.IGNORECASE):
            return True
    return False


def detect_multi_speaker_conversation(transcript_history: str, lookback: int = 6) -> bool:
    """
    Detect if multiple different speakers are conversing (not with AI).

    Args:
        transcript_history: The full conversation transcript
        lookback: Number of recent messages to analyze

    Returns:
        True if multiple speakers detected OR 3+ consecutive non-AI messages
    """
    lines = extract_last_replies(transcript_history, lookback)

    speakers = set()
    consecutive_non_ai = 0

    for line in lines:
        if line.startswith("AI:"):
            consecutive_non_ai = 0
        else:
            consecutive_non_ai += 1
            # Extract speaker ID (User, Speaker_0, Speaker_1, etc.)
            speaker = line.split(":")[0].strip()
            if speaker != "AI":
                speakers.add(speaker)

    # Multiple speakers detected OR 3+ consecutive non-AI messages
    return len(speakers) >= 2 or consecutive_non_ai >= 3


def calculate_response_delay(transcript: str, conversation: str) -> float:
    """
    Calculate dynamic delay before AI responds based on context.

    Args:
        transcript: The latest transcribed speech
        conversation: The full conversation history

    Returns:
        Delay in seconds before AI should respond
    """
    if is_directed_at_ai(transcript):
        return 0.0  # Respond immediately to direct questions
    if detect_multi_speaker_conversation(conversation):
        return 3.0  # Wait for user-to-user conversations
    return 0.5  # Default short delay

