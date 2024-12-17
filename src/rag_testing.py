import os
import re
from typing import Any, Dict, List, Literal, Optional
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.llama_api import LlamaAPI
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from tqdm import tqdm
import random


from transformers import AutoTokenizer
import torch
from llama_index.llms.huggingface import HuggingFaceLLM


load_dotenv(os.path.join(os.path.dirname(__file__), "../config/.env"))
os.environ["hf_token"] = os.getenv('hf_token')
hf_token = os.getenv('hf_token')


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=hf_token,
)

stopping_ids = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]




class QAPair(BaseModel):
    question: str = Field(description="A factoid question generated from the index.")
    answer: str = Field(description="Answer to factoid question.")

class QAGenerator:
    def __init__(self,
                model_name: str="meta-llama/Meta-Llama-3-8B-Instruct",
                temperature: float=0.3):
        # self.llm = HuggingFaceLLM(
        #         model_name=model_name,
        #         model_kwargs={
        #             "token": hf_token,
        #             "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
        #             # "quantization_config": quantization_config
        #         },
        #         generate_kwargs={
        #             "do_sample": True,
        #             "temperature": temperature,
        #             "top_p": 0.9,
        #         },
        #         tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
        #         tokenizer_kwargs={"token": hf_token},
        #         stopping_ids=stopping_ids,
        #                     ) ### UN-COMMENT IF USING LLAMA ###
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=400,
            strict=True
        )
        
        self.prompt_template = PromptTemplate(
            template="""
                    Your task is to write a factoid question and an answer given a context.
                    Your factoid question should be answerable with a specific, concise piece of factual information from the context.
                    Your factoid question should be formulated in the same style as questions users could ask in a search engine.
                    This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

                    Provide your answer as follows:

                    Output:::
                    Factoid question: (your factoid question)
                    Answer: (your answer to the factoid question)

                    Now here is the context.

                    Context: {context}\n
                    """
        )

        self.tool = FunctionTool.from_defaults(
            fn=self.structure_qa_pair,
            name="qa_pair"
        )
    
    @staticmethod
    def structure_qa_pair(question: str, answer: str) -> QAPair:
        """
        Creates and returns a structured QAPair object.
        """
        return QAPair(question=question, answer=answer)
    
    def create_pairs(self,
                    context: str,
                    ) -> str:
        

        prompt = self.prompt_template.format(context=context)


        messages = [
            ChatMessage(
                role="system", content="You are a helpful assistant that generates factoid Q&A pairs."
            ),
            ChatMessage(role="user", content=prompt),
        ]

        # Call the LLM with predict_and_call
        response = self.llm.predict_and_call(
            [self.tool],
            messages=messages
        )
        formatted_response = response.response.replace(" answer=", ", answer=")
        response_dict = eval(f"dict({formatted_response})")
        return response_dict

def create_qa_dataset(
        index: list,
        n: int=10
):
    print(f"Generating {n} QA pairs...")

    outputs = []

    for sampled_context in tqdm(random.sample(index, n)):
        try:
            generation = QAGenerator().create_pairs(sampled_context)
            question = generation["question"]
            answer = generation["answer"]

            outputs.append(
                {
                    "context": sampled_context.page_content,
                    "question": question,
                    "answer": answer,
                    "source_doc": sampled_context.metadata["source"]
                }
            )
        except Exception as e:
            print(f"Error creating QA pair {n}: {e}")
            continue

    return outputs


### GROUNDEDNESS GRADING OF QA PAIRS

class GroundednessScore(BaseModel):
            score: int = Field(ge=1, le=5, description="The groundedness score for the QA pair.")
            Evaluation: str = Field(description="A written justification for the score provided.")

class GroundednessGrader:
    def __init__(self,
        model_name: str="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature: float=0.3):

        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=400,
            strict=True
        )
        
        self.prompt_template = PromptTemplate(
            template="""
            You will be given a context and a question.
            Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
            Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

            Use the "GroundednessScore" tool to provide your response. 

            Call the tool with the following format:
            GroundednessScore(score=<rating>, Evaluation="<your rationale for the rating>")

            Context: {context}
            Question: {question}

            """
        )

        
        self.tool = FunctionTool.from_defaults(
            fn=self.grade_groundedness_function,
            name="GroundednessScore"
        )
    
    @staticmethod
    def grade_groundedness_function(score: int, Evaluation: str) -> GroundednessScore:
        """Grades the LLM-generated answer to determine if it is grounded in the provided documents."""
        return GroundednessScore(score=score, Evaluation=Evaluation)    
    
    def score_pair(self, question: str, context: str):
        try:
            prompt = self.prompt_template.format(context=context,
                                                question=question)

            messages = [
                ChatMessage(
                    role="system", content="You are a helpful assistant that grades factoid Q&A pairs."
                ),
                ChatMessage(role="user", content=prompt),
            ]

            response = self.llm.predict_and_call([self.tool], messages=messages)
            raw_output = response.response  # This is the plain string output
            print("Raw LLM Output:", raw_output)

            # Parse the raw output string
            match = re.search(r"score\s*=\s*(\d+)\s+Evaluation\s*=\s*['\"]?(.+?)['\"]?$", raw_output, re.DOTALL)
            if not match:
                raise ValueError("Response format is invalid. Expected 'score=<int> Evaluation='<text>'")

            score = int(match.group(1))
            evaluation = match.group(2)

            # Create and return the GroundednessScore object
            return GroundednessScore(score=score, Evaluation=evaluation)

        except Exception as e:
            print(f"Error processing QA pair: {e}")
            return None

class StandAloneScore(BaseModel):
    score: int = Field(ge=1, le=5, description="How well the question can stand alone.")
    Evaluation: str = Field(description="A written justification for the score provided.")

class StandaloneGrader:
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", temperature: float = 0.3):
        self.llm = OpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            max_tokens=400,
            strict=True
        )

        self.prompt_template = PromptTemplate(
            template="""
            You will be given a question.
            Your task is to evaluate how well the question can stand alone without needing additional context.

            Provide your answer on a scale of 1 to 5:
            - 1 means the question cannot stand alone and requires significant additional context.
            - 5 means the question is completely self-contained and does not require additional context.

            Use the "StandAloneScore" tool to provide your response.

            Call the tool with the following format:
            StandAloneScore(score=<rating>, Evaluation="<your rationale for the rating>")

            Question: {question}
            """
        )

        self.tool = FunctionTool.from_defaults(
            fn=self.grade_standalone_function,
            name="StandAloneScore"
        )

    @staticmethod
    def grade_standalone_function(score: int, Evaluation: str) -> StandAloneScore:
        """
        Grades how well the question stands alone.
        """
        return StandAloneScore(score=score, Evaluation=Evaluation)

    def score_question(self, question: str) -> StandAloneScore:
        """
        Evaluates the question for standalone quality and returns a StandAloneScore object.

        Args:
            question (str): The question to evaluate.

        Returns:
            StandAloneScore: A structured score and justification.
        """
        try:
            # Format the prompt
            prompt = self.prompt_template.format(question=question)

            messages = [
                ChatMessage(role="system", content="You are a helpful assistant that evaluates standalone questions."),
                ChatMessage(role="user", content=prompt),
            ]

            # Call the LLM with predict_and_call
            response = self.llm.predict_and_call([self.tool], messages=messages)
            raw_output = response.response  # Raw response as a string
            print("Raw LLM Output:", raw_output)

            # Parse the raw output into a structured form
            
            match = re.match(r"score=(\d+)\s+Evaluation='(.+)'", raw_output)
            if not match:
                raise ValueError("Response format is invalid. Expected 'score=<int> Evaluation='<text>'")

            # Extract values
            score = int(match.group(1))
            evaluation = match.group(2)

            # Return the structured StandAloneScore object
            return StandAloneScore(score=score, Evaluation=evaluation)

        except Exception as e:
            print(f"Error processing question: {e}")
            return None