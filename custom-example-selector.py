from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Optional


class CustomFewShotPromptTemplate:
    def __init__(self, examples, model_name="sbintuitions/sarashina-embedding-v1-1b"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'})

        self.example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template="問題: {input} -> 回答例: {output}",
        )
        self.example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            self.embeddings,
            Chroma,
            k=1)

        self.few_shot_prompt = FewShotPromptTemplate(
            example_selector=self.example_selector,
            example_prompt=self.example_prompt,
            input_variables=["input"],
            example_separator="\n\n",
            template_format="jinja2",
            suffix="" # This is an example suffix, adjust as needed
            )

    def format(self, input: str) -> str:
        return self.few_shot_prompt.format(input=input)
