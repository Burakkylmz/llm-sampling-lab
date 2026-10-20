
from unittest.mock import MagicMock

import pytest

from llm_sampling_lab.groq_client import generate_response


def test_generate_response_with_sampling_parameters():
    """Sampling parameters should be forwarded to Groq."""

    client = MagicMock()

    mock_response = client.chat.completions.create.return_value
    mock_response.choices[0].message.content = "Test slogan"

    result = generate_response(
        client=client,
        model="test-model",
        prompt="Generate a slogan.",
        temperature=0.7,
        top_p=0.9,
    )

    assert result == "Test slogan"

    client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "Generate a slogan.",
            }
        ],
        temperature=0.7,
        top_p=0.9,
        reasoning_effort="low",
        include_reasoning=False,
        max_completion_tokens=256,
    )


def test_generate_response_default_parameters():
    """Default sampling parameters should be 1.0."""

    client = MagicMock()

    mock_response = client.chat.completions.create.return_value
    mock_response.choices[0].message.content = "Default response"

    generate_response(
        client=client,
        model="test-model",
        prompt="Hello",
    )

    request_kwargs = (
        client.chat.completions.create.call_args.kwargs
    )

    assert request_kwargs["temperature"] == 1.0
    assert request_kwargs["top_p"] == 1.0


def test_generate_response_empty_content():
    """An empty model response should raise ValueError."""

    client = MagicMock()

    mock_response = client.chat.completions.create.return_value
    mock_response.choices[0].message.content = None

    with pytest.raises(
        ValueError,
        match="Groq returned an empty response",
    ):
        generate_response(
            client=client,
            model="test-model",
            prompt="Hello",
        )
