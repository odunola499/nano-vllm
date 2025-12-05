import trio
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class AsyncLLMEngine:

    def __init__(self, model, **kwargs):

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

        self.send_request_channel, self.receive_request_channel = trio.open_memory_channel(1000)
        self._batch_loop_started = False
        self._shutdown_event = trio.Event()



    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    async def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        send_request_output_channel, receive_request_output_channel = trio.open_memory_channel(1)
        seq.send_channel = send_request_output_channel
        await self.send_request_channel.send(seq)
        return receive_request_output_channel


    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids, seq.send_channel) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    async def background_loop(
            self
    ):
        while True:
            if self._shutdown_event.is_set():
                break
            try:
                while True:
                    seq = self.receive_request_channel.receive_nowait()
                    self.scheduler.add(seq)
            except trio.WouldBlock:
                pass
            if not self.scheduler.is_finished():
                outputs, num_tokens = await trio.to_thread.run_sync(self.step)
                for output in outputs:
                    seq_id, completion_token_ids, send_output_channel = output
                    output_text = self.tokenizer.decode(completion_token_ids)
                    await send_output_channel.send((seq_id, completion_token_ids, output_text))
            else:
                with trio.move_on_after(0.1):
                    seq = await self.receive_request_channel.receive()
                    self.scheduler.add(seq)


    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ):
        output_channel = await self.add_request(
            prompt,sampling_params
        )
        seq_id, completion_token_ids, output_text = await output_channel.receive()
        outputs = {'text':output_text, 'token_ids':completion_token_ids}
        return outputs

    async def shutdown(self):
        self._shutdown_event.set()



