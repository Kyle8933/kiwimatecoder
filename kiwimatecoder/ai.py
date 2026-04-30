import httpx
from rich.console import Console

console = Console()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "mistralai/devstral-2512"


async def stream_response(prompt: str, api_key: str, model: str = DEFAULT_MODEL):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kiwimatecoder.com",
        "X-Title": "KiwiMateCoder",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are KiwiMateCoder, an expert coding assistant. Give clear, concise, and accurate coding help. Prefer showing code over lengthy explanations.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST", OPENROUTER_URL, json=payload, headers=headers
        ) as response:
            if response.status_code != 200:
                console.print(f"[red]Error: {response.status_code}[/red]")
                return

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json

                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            console.print(token, end="")
                    except Exception:
                        continue
