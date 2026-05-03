from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable


@dataclass(frozen=True)
class PromptJob:
    name: str
    prompt: str
    meta: dict[str, Any] = field(default_factory=dict)


async def _call_client(client: Callable[[str], Any], prompt: str) -> Any:
    result = client(prompt)
    if inspect.isawaitable(result):
        return await result
    return await asyncio.to_thread(lambda: result) if callable(client) else result


async def batch_prompts_async(
    jobs: Iterable[PromptJob],
    *,
    client: Callable[[str], Any],
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Execute multiple prompt calls concurrently.

    The client may be sync or async. Results are returned by job name.
    """
    jobs_list = list(jobs)
    if not jobs_list:
        return {}

    sem = asyncio.Semaphore(max(1, int(max_concurrency or 1)))

    async def _worker(job: PromptJob) -> tuple[str, Any]:
        async with sem:
            result = client(job.prompt)
            if inspect.isawaitable(result):
                result = await result
            else:
                result = await asyncio.to_thread(lambda: result)
            return job.name, result

    results = await asyncio.gather(*(_worker(job) for job in jobs_list))
    return {name: result for name, result in results}


def batch_prompts_sync(
    jobs: Iterable[PromptJob],
    *,
    client: Callable[[str], Any],
    max_concurrency: int = 4,
) -> dict[str, Any]:
    """Synchronous wrapper around batch_prompts_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Safe fallback when already inside an event loop.
        return asyncio.get_event_loop().run_until_complete(
            batch_prompts_async(jobs, client=client, max_concurrency=max_concurrency)
        )

    return asyncio.run(batch_prompts_async(jobs, client=client, max_concurrency=max_concurrency))
