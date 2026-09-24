"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import os

import requests
from langchain import LLMChain, OpenAI, PromptTemplate

FASTCHAT_API_BASE = os.environ.get("FASTCHAT_API_BASE", "http://localhost:8888/v1")
FASTCHAT_WORKER = os.environ.get("FASTCHAT_WORKER", "http://localhost:31022")


class FastChatLLM:
    """One agent role, served as a LoRA adapter by the FastChat multi-model worker."""

    def __init__(self, model_name):
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["OPENAI_API_BASE"] = FASTCHAT_API_BASE
        self.model_name = model_name
        self.prompt = PromptTemplate(input_variables=["prompt"], template="{prompt}")

    def run(self, prompt, temperature=0.9, stop=["\n"], max_tokens=128):
        llm = OpenAI(
            model=self.model_name,
            temperature=temperature,
            top_p=0.75,
            stop=stop,
            max_tokens=max_tokens,
            cache=False,
            model_kwargs={"top_k": 40, "num_beams": 4},
        )
        try:
            return LLMChain(llm=llm, prompt=self.prompt).run(prompt)
        except Exception as e:
            print(e)
            return ""


def get_llm_backend(model_name):
    return FastChatLLM(model_name)


def reset_kv_cache():
    """Drop the trajectory's shared KV cache on the worker before the next question."""
    requests.post(f"{FASTCHAT_WORKER}/precache_reset", timeout=60).raise_for_status()
