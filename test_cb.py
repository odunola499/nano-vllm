import os
import trio
import random
from nanovllm import AsyncLLM, SamplingParams


async def run_single_request(llm, prompt, idx):
    params = SamplingParams(max_tokens=32)
    print(f"[{idx}] sending request: {prompt}")
    out = await llm.generate(prompt, params)
    print(f"[{idx}] done: {out['text']!r}")
    return out


async def main():
    random.seed(0)

    model_path = os.path.expanduser("huggingface/Qwen3-0.6B/")
    llm = AsyncLLM(model_path, enforce_eager=False, max_model_len=4096)

    print('Model loaded. Starting concurrent tests...\n')

    prompts = [
        "Superman is quite a ",
        "But why should we go to school when",
        "The rise and fall of the roman empire is",
        "The yoruba ",
        "Trio concurrency",
        "The country that spends money",
    ]

    async with trio.open_nursery() as nursery:
        nursery.start_soon(llm.background_loop)

        await trio.sleep(0.1)

        for idx, prompt in enumerate(prompts):
            delay = random.uniform(0, 1.5)
            nursery.start_soon(run_single_request, llm, prompt, idx)
            await trio.sleep(delay)

    print("Done")


if __name__ == "__main__":
    trio.run(main)