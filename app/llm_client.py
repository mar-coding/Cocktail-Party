"""
LLM Client Module
-----------------
Contains all payload definitions and functions for interacting with LLM providers.
Supports multiple providers: Ollama (default) and vLLM (RunPod).
"""

import os
import requests
import json
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =============================================================================
# LLM Provider Configuration
# =============================================================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # "ollama" or "vllm"

# vLLM (RunPod) settings
VLLM_API_TOKEN = os.getenv("VLLM_API_TOKEN", "")
VLLM_API_ADDRESS = os.getenv("VLLM_API_ADDRESS", "")  # Full base URL
VLLM_MODEL = os.getenv("VLLM_MODEL", "google/gemma-3-4b-it")

# =============================================================================
# Connection Pooling for Reduced Latency
# =============================================================================
_session = None


def get_session():
    """Get a persistent session with connection pooling for reduced latency."""
    global _session
    if _session is None:
        _session = requests.Session()
        # Configure retry strategy - only retry on connection errors, not timeouts
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],  # Retry on server errors
            allowed_methods=["POST"],  # Allow retry on POST for Ollama
            raise_on_status=False,  # Don't raise on retries, let us handle errors
        )
        adapter = HTTPAdapter(
            pool_connections=5, pool_maxsize=5, max_retries=retry_strategy
        )
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


# =============================================================================
# vLLM Client (Lazy Initialization)
# =============================================================================
_vllm_client = None


def get_vllm_client():
    """Get a lazy-initialized vLLM (OpenAI-compatible) client."""
    global _vllm_client
    if _vllm_client is None and VLLM_API_TOKEN and VLLM_API_ADDRESS:
        try:
            from openai import OpenAI

            _vllm_client = OpenAI(api_key=VLLM_API_TOKEN, base_url=VLLM_API_ADDRESS)
            logger.info(
                f"vLLM client initialized: model={VLLM_MODEL}, endpoint={VLLM_API_ADDRESS}"
            )
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
    return _vllm_client


# Production mode flag
IS_PROD = os.getenv("IS_PROD", "false").lower() == "true"

# HTTP request timeouts (seconds)
# Connect timeout: how long to wait for initial connection
HTTP_CONNECT_TIMEOUT = int(os.getenv("HTTP_CONNECT_TIMEOUT", "10"))
# Read timeout: how long to wait for data (longer for LLM inference)
HTTP_READ_TIMEOUT = int(os.getenv("HTTP_READ_TIMEOUT", "120"))
# Combined timeout tuple for requests
HTTP_TIMEOUT = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")


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

CONTRIBUTION_PROMPT = (
    "You are at a Cocktail Party. You've been listening to guests converse. "
    "Contribute a brief, interesting fact or comment related to their discussion. "
    "Be concise (1-2 sentences). Start naturally like 'Speaking of that...' or just jump in. "
    "Output will be audio - no emojis or special characters."
)

# ============================================================================
# PAYLOAD BUILDERS
# ============================================================================


def build_chat_payload(
    user_content: str,
    system_prompt: str = COCKTAIL_PARTY_PROMPT,
    model: str = None,
    num_predict: int = 100,
    stream: bool = True,
) -> dict:
    """Build a chat payload for Ollama API."""
    return {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "options": {"num_predict": num_predict},
        "keep_alive": "30m",
        "stream": stream,
    }


# ============================================================================
# LLM FUNCTIONS
# ============================================================================


def _stream_ollama_response(transcript: str, system_prompt: str):
    """
    Stream response from Ollama API, yielding raw tokens.
    Phrase buffering for TTS is handled downstream by talk().

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: The system prompt to use

    Yields:
        str: Raw text tokens as they arrive from the API

    Raises:
        LLMError: If there's an error communicating with Ollama
    """
    payload = build_chat_payload(transcript, system_prompt=system_prompt)

    try:
        with get_session().post(
            OLLAMA_URL, json=payload, stream=True, timeout=HTTP_TIMEOUT
        ) as r:
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
                    if (
                        "message" in data
                        and isinstance(data["message"], dict)
                        and "content" in data["message"]
                    ):
                        chunk = data["message"]["content"].replace("*", "")
                    elif "delta" in data:
                        chunk = str(data["delta"]).replace("*", "")

                if chunk:
                    yield chunk
    except requests.exceptions.ReadTimeout:
        logger.error(
            f"LLM read timed out after {HTTP_READ_TIMEOUT}s - model may be slow or overloaded"
        )
        raise LLMError("LLM response timed out - try increasing HTTP_READ_TIMEOUT")
    except requests.exceptions.ConnectTimeout:
        logger.error(f"Failed to connect to LLM service within {HTTP_CONNECT_TIMEOUT}s")
        raise LLMError("Connection to LLM service timed out - is Ollama running?")
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        raise LLMError("Request to LLM service timed out")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to LLM service: {e}")
        raise LLMError("Failed to connect to LLM service - is Ollama running?")
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response else "No response body"
        logger.error(
            f"LLM service returned an error: {e.response.status_code} - {error_body}"
        )
        raise LLMError(f"LLM service returned an error: {error_body}")


def _stream_vllm_response(transcript: str, system_prompt: str, max_tokens: int = 250):
    """
    Stream response from vLLM (OpenAI-compatible) API, yielding raw tokens.

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: The system prompt to use
        max_tokens: Maximum tokens to generate

    Yields:
        str: Raw text tokens as they arrive from the API

    Raises:
        LLMError: If there's an error communicating with the vLLM API
    """
    client = get_vllm_client()
    if client is None:
        raise LLMError(
            "vLLM client not initialized - check VLLM_API_TOKEN and VLLM_API_ADDRESS"
        )

    try:
        stream = client.chat.completions.create(
            model=VLLM_MODEL,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["User:", "AI:", "\nUser", "\nAI"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content.replace("*", "")
                if text:
                    yield text
    except Exception as e:
        logger.error(f"vLLM API error: {e}")
        raise LLMError(f"vLLM API error: {e}")


def _get_vllm_response(
    transcript: str, system_prompt: str, max_tokens: int = 150
) -> str:
    """
    Get a complete (non-streaming) response from vLLM (OpenAI-compatible) API.

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: The system prompt to use
        max_tokens: Maximum tokens to generate

    Returns:
        str: The complete response text

    Raises:
        LLMError: If there's an error communicating with the vLLM API
    """
    client = get_vllm_client()
    if client is None:
        raise LLMError(
            "vLLM client not initialized - check VLLM_API_TOKEN and VLLM_API_ADDRESS"
        )

    try:
        response = client.chat.completions.create(
            model=VLLM_MODEL,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["User:", "AI:", "\nUser", "\nAI"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.replace("*", "")
        return ""
    except Exception as e:
        logger.error(f"vLLM API error: {e}")
        raise LLMError(f"vLLM API error: {e}")


def stream_llm_response(
    transcript: str,
    system_prompt: str = COCKTAIL_PARTY_PROMPT,
    alone: bool = False,
    is_back_and_forth: bool = False,
    contribution_mode: bool = False,
):
    """
    Streams text chunks from the configured LLM provider.
    Routes to vLLM or Ollama based on LLM_PROVIDER setting.

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: Optional custom system prompt (defaults to COCKTAIL_PARTY_PROMPT)
        alone: Whether AI has been talking alone (no user responses)
        is_back_and_forth: Whether there's an active back-and-forth conversation
        contribution_mode: Whether to use contribution prompt (for user-to-user conversations)

    Yields:
        str: Text chunks as they arrive from the LLM

    Raises:
        LLMError: If there's an error communicating with the LLM service
    """
    # Select appropriate prompt based on mode
    if contribution_mode:
        selected_prompt = CONTRIBUTION_PROMPT
    elif alone:
        selected_prompt = ALONE_PROMPT
    elif is_back_and_forth:
        selected_prompt = BACK_AND_FORTH_PROMPT
    else:
        selected_prompt = system_prompt

    # Route to appropriate provider
    if LLM_PROVIDER == "vllm":
        yield from _stream_vllm_response(transcript, selected_prompt)
    else:
        yield from _stream_ollama_response(transcript, selected_prompt)


def get_llm_response(
    transcript: str, system_prompt: str = SUMMARY_PROMPT, summarize: bool = True
) -> str:
    """
    Gets a complete (non-streaming) response from the configured LLM provider.
    Routes to vLLM or Ollama based on LLM_PROVIDER setting.

    Args:
        transcript: The conversation transcript to send to the LLM
        system_prompt: Optional custom system prompt

    Returns:
        str: The complete response text

    Raises:
        LLMError: If there's an error communicating with the LLM service
    """
    # Route to appropriate provider
    if LLM_PROVIDER == "vllm":
        return _get_vllm_response(transcript, system_prompt)

    # Ollama path
    payload = build_chat_payload(transcript, system_prompt=system_prompt, stream=False)

    # Only log payload in development mode
    if not IS_PROD:
        logger.debug(f"LLM payload: {payload}")

    try:
        response = get_session().post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.ReadTimeout:
        logger.error(
            f"LLM read timed out after {HTTP_READ_TIMEOUT}s - model may be slow or overloaded"
        )
        raise LLMError("LLM response timed out - try increasing HTTP_READ_TIMEOUT")
    except requests.exceptions.ConnectTimeout:
        logger.error(f"Failed to connect to LLM service within {HTTP_CONNECT_TIMEOUT}s")
        raise LLMError("Connection to LLM service timed out - is Ollama running?")
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        raise LLMError("Request to LLM service timed out")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to LLM service: {e}")
        raise LLMError("Failed to connect to LLM service - is Ollama running?")
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response else "No response body"
        logger.error(
            f"LLM service returned an error: {e.response.status_code} - {error_body}"
        )
        raise LLMError(f"LLM service returned an error: {error_body}")

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise LLMError("Invalid response from LLM service")

    # Validate response structure
    if not isinstance(data, dict):
        logger.error("LLM response is not a valid JSON object")
        raise LLMError("Invalid response format from LLM service")

    if (
        "message" in data
        and isinstance(data["message"], dict)
        and "content" in data["message"]
    ):
        return data["message"]["content"].replace("*", "")
    return ""


def stream_custom_chat(
    user_content: str,
    system_prompt: str,
    model: str = None,
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
        with get_session().post(
            OLLAMA_URL, json=payload, stream=True, timeout=HTTP_TIMEOUT
        ) as r:
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
                    if (
                        "message" in data
                        and isinstance(data["message"], dict)
                        and "content" in data["message"]
                    ):
                        chunk = data["message"]["content"].replace("*", "")
                    elif "delta" in data:
                        chunk = str(data["delta"]).replace("*", "")

                if chunk:
                    yield chunk
    except requests.exceptions.ReadTimeout:
        logger.error(
            f"LLM read timed out after {HTTP_READ_TIMEOUT}s - model may be slow or overloaded"
        )
        raise LLMError("LLM response timed out - try increasing HTTP_READ_TIMEOUT")
    except requests.exceptions.ConnectTimeout:
        logger.error(f"Failed to connect to LLM service within {HTTP_CONNECT_TIMEOUT}s")
        raise LLMError("Connection to LLM service timed out - is Ollama running?")
    except requests.exceptions.Timeout:
        logger.error("LLM request timed out")
        raise LLMError("Request to LLM service timed out")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to LLM service: {e}")
        raise LLMError("Failed to connect to LLM service - is Ollama running?")
    except requests.exceptions.HTTPError as e:
        error_body = e.response.text if e.response else "No response body"
        logger.error(
            f"LLM service returned an error: {e.response.status_code} - {error_body}"
        )
        raise LLMError(f"LLM service returned an error: {error_body}")
