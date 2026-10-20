
import os

from groq import Groq


def create_groq_client() -> Groq:
    """Create a Groq client using the environment API key."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key or not api_key.strip():
        raise ValueError(
            "GROQ_API_KEY environment variable is missing or empty."
        )

    return Groq(
        api_key=api_key,
        timeout=30.0,
        max_retries=0,
    )



def generate_response(
    client: Groq,
    model: str,
    prompt: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """Generate a response with configurable sampling parameters."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temperature,
        top_p=top_p,
        reasoning_effort="low",
        include_reasoning=False,
        max_completion_tokens=256,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "Groq returned an empty response."
        )

    return content
