"""
LLM Client Module
-----------------
Contains all payload definitions and functions for interacting with the local Ollama instance.
"""

import os
import requests
import json
from loguru import logger

# Production mode flag
IS_PROD = os.getenv("IS_PROD", "false").lower() == "true"

# HTTP request timeout (seconds)
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

COCKTAIL_PARTY_PROMPT = (
    "You are a LLM in a WebRTC call simulating a Cocktail Party. Your goal is to "
    "be chill and answer in a cool way be super interesting and engaging. The "
    "output will be converted to audio so don't include emojis "
    "or special characters in your answers. Respond to what the "
    "user said in a creative way base yourself off of the conversation transcript in which AI represents you, "
    "User represents the User you have to reply to. "
    "DONT ANSWER WITH Ai:or User:, directly speak what you need to speak."
)
ALONE_PROMPT = (
    "You are a LLM in a WebRTC call simulating a Cocktail Party."
    "The output will be converted to audio so don't include emojis "
    "or special characters in your answers. Don't start the answer with 'AI:' just speak." 
    "Right now no one answered your last 2 replies, so you need to"
    "Answer with a way to get the user to respond." 
    "You should try to be funny and engaging to get the user to respond."
    "We included your last 2 replies in the conversation transcript"
    "for which you havent received a reply."
)

SUMMARY_PROMPT = (
    "You are a summarization assistant. The user message contains a conversation transcript. "
    "Summarize it into a concise summary of no more than 100 words. "
    "Only output the summary text directly, no preamble or labels. "
    "If the transcript is empty or has nothing meaningful, respond with an empty string."
)

BACK_AND_FORTH_PROMPT = (
    "You are a LLM in a WebRTC call simulating a Cocktail Party. "
    "Your goal is to be interesting. The "
    "output will be converted to audio so don't include emojis "
    "or special characters in your answers."
    "In the conversation transcript: AI represents you, "
    "User represents the User you have to reply to. "
    "DONT ANSWER WITH Ai: or User:, directly speak what you need to speak."
    "try to comment on the perceived mood of User's replies while answering him/the subject of the conversation."
)

# ============================================================================
# PAYLOAD BUILDERS
# ============================================================================

def build_chat_payload(
    user_content: str,
    system_prompt: str = COCKTAIL_PARTY_PROMPT,
    model: str = "gemma3:4b",
    num_predict: int = 100,
    stream: bool = True,
) -> dict:
    """Build a chat payload for Ollama API."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "options": {"num_predict": num_predict},
        "stream": stream,
    }


# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def stream_llm_response(transcript: str, system_prompt: str = COCKTAIL_PARTY_PROMPT, alone: bool = False, is_back_and_forth: bool = False):
    """
    Streams text chunks from Ollama /api/chat with stream=true.
    Yields small pieces of text as they come.

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: Optional custom system prompt (defaults to COCKTAIL_PARTY_PROMPT)

    Yields:
        str: Text chunks as they arrive from the LLM

    Raises:
        LLMError: If there's an error communicating with the LLM service
    """
    payload = build_chat_payload(transcript, system_prompt=ALONE_PROMPT if alone else BACK_AND_FORTH_PROMPT if is_back_and_forth else system_prompt)

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=HTTP_TIMEOUT) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response line: {e}")
                    continue

                chunk = ""
                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                        chunk = data["message"]["content"].replace("*", "")
                    elif "delta" in data:
                        chunk = str(data["delta"]).replace("*", "")

                if chunk:
                    yield chunk
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        raise LLMError("Request to LLM service timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to LLM service")
        raise LLMError("Failed to connect to LLM service")
    except requests.exceptions.HTTPError as e:
        logger.error(f"LLM service returned an error: {e.response.status_code}")
        raise LLMError("LLM service returned an error")


def get_llm_response(transcript: str, system_prompt: str = SUMMARY_PROMPT, summarize: bool = True) -> str:
    """
    Gets a complete (non-streaming) response from Ollama.

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: Optional custom system prompt

    Returns:
        str: The complete response text

    Raises:
        LLMError: If there's an error communicating with the LLM service
    """
    payload = build_chat_payload(transcript, system_prompt=system_prompt, stream=False)

    # Only log payload in development mode
    if not IS_PROD:
        logger.debug(f"LLM payload: {payload}")

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        raise LLMError("Request to LLM service timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to LLM service")
        raise LLMError("Failed to connect to LLM service")
    except requests.exceptions.HTTPError as e:
        logger.error(f"LLM service returned an error: {e.response.status_code}")
        raise LLMError("LLM service returned an error")

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise LLMError("Invalid response from LLM service")

    # Validate response structure
    if not isinstance(data, dict):
        logger.error("LLM response is not a valid JSON object")
        raise LLMError("Invalid response format from LLM service")

    if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
        return data["message"]["content"].replace("*", "")
    return ""


def stream_custom_chat(
    user_content: str,
    system_prompt: str,
    model: str = "gemma3:4b",
    num_predict: int = 600,
):
    """
    Stream a custom chat with full control over parameters.

    Args:
        user_content: The user message content
        system_prompt: The system prompt to use
        model: The Ollama model to use
        num_predict: Max tokens to generate

    Yields:
        str: Text chunks as they arrive

    Raises:
        LLMError: If there's an error communicating with the LLM service
    """
    payload = build_chat_payload(
        user_content,
        system_prompt=system_prompt,
        model=model,
        num_predict=num_predict,
        stream=True,
    )

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=HTTP_TIMEOUT) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response line: {e}")
                    continue

                chunk = ""
                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                        chunk = data["message"]["content"].replace("*", "")
                    elif "delta" in data:
                        chunk = str(data["delta"]).replace("*", "")

                if chunk:
                    yield chunk
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        raise LLMError("Request to LLM service timed out")
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to LLM service")
        raise LLMError("Failed to connect to LLM service")
    except requests.exceptions.HTTPError as e:
        logger.error(f"LLM service returned an error: {e.response.status_code}")
        raise LLMError("LLM service returned an error")

