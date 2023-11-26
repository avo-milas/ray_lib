from starlette.requests import Request

import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

from transformers import pipeline


@serve.deployment
class Translator:
    def __init__(self):
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    def translate(self, text: str) -> str:
        model_output = self.model(text)
        translation = model_output[0]["translation_text"]
        return translation


@serve.deployment
class Summarizer:
    def __init__(self, translator):
        self.model = pipeline("summarization", model="t5-small")
        self.translator: DeploymentHandle = translator.options(use_new_handle_api=True)

    def summarize(self, text: str) -> str:
        model_output = self.model(text, min_length=5, max_length=15)
        summary = model_output[0]["summary_text"]

        return summary

    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        summary = self.summarize(english_text)

        translation = await self.translator.translate.remote(summary)
        return translation


app = Summarizer.bind(Translator.bind())
