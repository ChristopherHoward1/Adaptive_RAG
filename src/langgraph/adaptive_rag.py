import numpy as np
import os
import json

from typing import Literal, List, Any, Dict, Optional
from typing_extensions import TypedDict

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from IPython.display import Markdown, display

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from rank_bm25 import BM25Okapi
from pprint import pprint
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core import PromptTemplate
from llama_index.core.node_parser import TextSplitter

from helper_fns import process_file

load_dotenv(os.path.join(os.path.dirname(__file__), "../../config/.env"))
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Variables


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "no-retrieval"] = Field(
        ...,
        description="Given a user query, choose to route it to a vectorstore or to the LLM's built-in context.",
    )

class QueryAnalyzer:
    """
    Classifies the user query to determine if it requires retrieval from the knowledge base
    or can be answered directly by the LLM.
    """

    def __init__(
        self,
        knowledge_base_description: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        strict: bool = True
    ):
        # Initialize the LLM with function calling capabilities
        self.llm = OpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=4000,
            strict=strict
        )
        self.knowledge_base_description = knowledge_base_description

        # Define the system prompt
        self.system_prompt = (
            f"You are an expert at routing a user question to a vectorstore or to an LLM. "
            f"The vectorstore contains documents related to: {knowledge_base_description} "
            f"Use the vectorstore for questions on these topics. Otherwise, use no-retrieval."
        )

        # Create the FunctionTool instance inside the class
        self.tool = FunctionTool.from_defaults(fn=self.route_query)

    @staticmethod
    def route_query(datasource: str) -> RouteQuery:
        """Route a user query to the most relevant datasource."""
        return RouteQuery(datasource=datasource)

    def analyze_query(self, question: str) -> str:
        """
        Determines the query's relevance to the knowledge base.
        Returns 'vectorstore' or 'no-retrieval' based on the analysis.
        """
        # Create the message list using ChatMessage
        messages = [
            ChatMessage(
                role="system", content=self.system_prompt
            ),
            ChatMessage(role="user", content=question),
        ]

        # Call the LLM with predict_and_call
        response = self.llm.predict_and_call(
            [self.tool],
            prompt=messages
        )

        # The response should contain the function call with arguments
        # Extract the arguments from response.additional_kwargs['function_call']
        # if response.additional_kwargs and 'function_call' in response.additional_kwargs:
        #     function_call = response.additional_kwargs['function_call']
        #     if function_call.get('name') == 'route_query':
        #         arguments = function_call.get('arguments')
        #         # arguments is a JSON string
        #         args = json.loads(arguments)
        #         route_query_result = RouteQuery(**args)
        #         return route_query_result.datasource
        # else:
        #     # Handle unexpected response
        #     return "no-retrieval"
        return str(response)




### Query Transform

class StrOutputParser(BaseOutputParser):
    """Simple output parser that returns the text as is."""
    def parse(self, text: str) -> str:
        return text.strip()
    
# # class SubQueries(BaseModel):
#     sub_queries: List[str] = Field(description="List of sub-queries generated to help answer the user query.")

class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        ...,
        description="The rewritten query optimized for vectorstore retrieval."
    )

class StepBackQuery(BaseModel):
    stepback_query: str = Field(
        ...,
        description="A more general query to help retrieve relevant background information."
    )

class SubQueries(BaseModel):
    subqueries: List[str] = Field(
        ...,
        description="A list of 2-4 simpler sub-queries."
    )

class QueryTransformer:
    """
    Rewrites the input question to a better version optimized for vectorstore retrieval.
    Generates a step-back query to get broad context.
    Decomposes original query into simpler subqueries.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
    ):
        # Initialize the LLM
        self.llm = OpenAI(
            model=model_name,
            temperature=0,
            max_tokens=4000,
            strict=True
        )

        # Define the system prompts
        self.rewrite_system_prompt = (
            """You are an AI assistant that converts an input question to a better version 
            that is optimized for vectorstore retrieval. Look at the input and try to reason 
            about the underlying semantic intent/meaning."""
        )
        self.stepback_system_prompt = (
            """You are an AI assistant that takes input questions and generates more broad, 
            general queries to improve context retrieval in the RAG system.
            Given the original query, generate a step-back query that is more general and can help retrieve relevant background information."""
        )
        self.subquery_system_prompt = (
            """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
            Given the original query, decompose it into 2-4 distinct, simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

            Example: What TV shows should I watch?

            1. What are the top-rated shows across various streaming platforms?
            2. What shows are trending or highly recommended in popular genres (e.g., drama, comedy, thriller)?
            3. Are there any recently released shows that have received critical acclaim?
            4. What shows have won awards or received nominations in the past few years? 
            """
        )

        # Create FunctionTools for each function, specifying names
        self.rewrite_tool = FunctionTool.from_defaults(
            fn=self.rewrite_query_function,
            name="rewrite_query"
        )
        self.stepback_tool = FunctionTool.from_defaults(
            fn=self.stepback_query_function,
            name="stepback_query"
        )
        self.subquery_tool = FunctionTool.from_defaults(
            fn=self.decompose_question_function,
            name="decompose_question"
        )

    @staticmethod
    def rewrite_query_function(rewritten_query: str) -> RewrittenQuery:
        """Rewrites the input question to improve vectorstore retrieval performance."""
        return RewrittenQuery(rewritten_query=rewritten_query)

    @staticmethod
    def stepback_query_function(stepback_query: str) -> StepBackQuery:
        """Generates a more general step-back query based on the input question."""
        return StepBackQuery(stepback_query=stepback_query)

    @staticmethod
    def decompose_question_function(subqueries: List[str]) -> SubQueries:
        """Decomposes the input question into simpler sub-questions."""
        return SubQueries(subqueries=subqueries)

    def rewrite_question(self, question: str) -> str:
        """Rewrites the input question to improve vectorstore retrieval performance."""
        # Create the message list
        messages = [
            ChatMessage(role="system", content=self.rewrite_system_prompt),
            ChatMessage(
                role="user",
                content=f"Here is the initial question:\n\n{question}\n\nFormulate an improved question.",
            ),
        ]

        # Call the LLM with the rewrite_tool
        response = self.llm.predict_and_call(
            [self.rewrite_tool],
            prompt=messages,
            temperature=0,  # Override temperature if needed
        )

        # Parse the response
        if response.additional_kwargs and 'function_call' in response.additional_kwargs:
            function_call = response.additional_kwargs['function_call']
            if function_call.get('name') == 'rewrite_query':
                arguments = function_call.get('arguments')
                args = json.loads(arguments)
                rewritten_query_obj = RewrittenQuery(**args)
                return rewritten_query_obj.rewritten_query
        else:
            # Handle unexpected response
            print(f"Error processing question: {question}")
            return question  # Return the original question if transformation fails

    def generate_stepback_query(self, question: str) -> str:
        """Generates a more general step-back query based on the input question."""
        # Create the message list
        messages = [
            ChatMessage(role="system", content=self.stepback_system_prompt),
            ChatMessage(
                role="user",
                content=f"Here is the initial question:\n\n{question}\n\nFormulate a more general question that captures the semantic meaning of the original.",
            ),
        ]

        # Call the LLM with the stepback_tool
        response = self.llm.predict_and_call(
            [self.stepback_tool],
            prompt=messages,
            temperature=0,  # Override temperature if needed
        )

        # Parse the response
        if response.additional_kwargs and 'function_call' in response.additional_kwargs:
            function_call = response.additional_kwargs['function_call']
            if function_call.get('name') == 'stepback_query':
                arguments = function_call.get('arguments')
                args = json.loads(arguments)
                stepback_query_obj = StepBackQuery(**args)
                return stepback_query_obj.stepback_query
        else:
            # Handle unexpected response
            print(f"Error processing question: {question}")
            return question  # Return the original question if transformation fails

    def decompose_question(self, question: str) -> List[str]:
        """Decomposes the input question into simpler sub-questions."""
        # Create the message list
        messages = [
            ChatMessage(role="system", content=self.subquery_system_prompt),
            ChatMessage(
                role="user",
                content=f"Here is the initial question:\n\n{question}\n\nFormulate sub-questions that help answer the original question.",
            ),
        ]

        # Call the LLM with the subquery_tool
        response = self.llm.predict_and_call(
            [self.subquery_tool],
            prompt=messages,
            temperature=0.3,  # Override temperature for this call
        )

        # Parse the response
        if response.additional_kwargs and 'function_call' in response.additional_kwargs:
            function_call = response.additional_kwargs['function_call']
            if function_call.get('name') == 'decompose_question':
                arguments = function_call.get('arguments')
                args = json.loads(arguments)
                subqueries_obj = SubQueries(**args)
                subqueries = subqueries_obj.subqueries
                print("Decomposed Query:")
                for idx, subquery in enumerate(subqueries):
                    print(f"Sub-query {idx+1}: {subquery}")
                return subqueries
        else:
            # Handle unexpected response
            return []  # Return empty list if decomposition fails


### Answer Grader
class HallucinationGradingResult(BaseModel):
    """Model representing the hallucination grading result."""
    score: str = Field(
        description="A binary score 'yes' or 'no'. 'Yes' means the answer is grounded in and supported by the facts."
    )

class HallucinationGrader:
    """
    Grades an LLM-generated answer to determine if it is grounded in the provided documents.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 1000,
        strict: bool = True,
    ):
        # Initialize the LLM
        self.llm = OpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            strict=strict
        )

        # Define the system prompt
        self.system_prompt = """You are an evaluator determining if an LLM-generated response is based on or supported by a provided set of retrieved facts.
Provide a binary score: 'yes' or 'no'.
'Yes' means the answer is grounded in and supported by the facts."""

        # Create the FunctionTool instance inside the class
        self.tool = FunctionTool.from_defaults(
            fn=self.grade_hallucination_function,
            name="grade_hallucination"
        )

    @staticmethod
    def grade_hallucination_function(score: str) -> HallucinationGradingResult:
        """Grades the LLM-generated answer to determine if it is grounded in the provided documents."""
        return HallucinationGradingResult(score=score)

    def grade_hallucination(self, documents: str, generation: str) -> str:
        """
        Grades the LLM-generated answer to determine if it is grounded in the provided documents.
        Returns 'yes' or 'no' based on the grading.
        """
        try:
            # Create the message list
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(
                    role="user",
                    content=f"Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}\n\n",
                ),
            ]

            # Call the LLM with the tool
            response = self.llm.predict_and_call(
                [self.tool],
                prompt=messages,
            )

            # Parse the response
            if response.additional_kwargs and 'function_call' in response.additional_kwargs:
                function_call = response.additional_kwargs['function_call']
                if function_call.get('name') == 'grade_hallucination':
                    arguments = function_call.get('arguments')
                    args = json.loads(arguments)
                    grading_result = HallucinationGradingResult(**args)
                    return grading_result.score.lower()
            else:
                # Handle unexpected response
                return 'no'  # Default to 'no' if there is an error
        except Exception as e:
            print(f"Error grading hallucination: {e}")
            return 'no'  # Default to 'no' if there is an error



### Answer Grader

class GradeAnswer(BaseModel):
    """Binary score to assess if the answer addresses the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class AnswerGrader:
    """
    Grades an LLM-generated answer to determine if it adequately addresses the user's question.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 4000,
        strict: bool = True,
    ):
        # Initialize the LLM with function calling capabilities
        self.llm = OpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            strict=strict
        )

        # Define the system prompt
        self.system_prompt = """You are a grader assessing whether an answer addresses or resolves a question.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""

        # Create the FunctionTool instance inside the class
        self.tool = FunctionTool.from_defaults(
            fn=self.grade_answer_function,
            name="grade_answer"
        )

    @staticmethod
    def grade_answer_function(binary_score: str) -> GradeAnswer:
        """Grades the LLM-generated answer to determine if it resolves the user's question."""
        return GradeAnswer(binary_score=binary_score)

    def grade_answer(self, question: str, generation: str) -> str:
        """
        Grades the answer generated by the LLM to determine if it resolves the user's question.
        Returns 'yes' or 'no' based on the grading.
        """
        try:
            # Create the message list
            messages = [
                ChatMessage(role="system", content=self.system_prompt),
                ChatMessage(
                    role="user",
                    content=f"User question:\n\n{question}\n\nLLM generation:\n\n{generation}"
                ),
            ]

            # Call the LLM with the tool
            response = self.llm.predict_and_call(
                [self.tool],
                prompt=messages,
            )

            # Parse the response
            if response.additional_kwargs and 'function_call' in response.additional_kwargs:
                function_call = response.additional_kwargs['function_call']
                if function_call.get('name') == 'grade_answer':
                    arguments = function_call.get('arguments')
                    # arguments is a JSON string
                    args = json.loads(arguments)
                    grading_result = GradeAnswer(**args)
                    return grading_result.binary_score.lower()
            else:
                # Handle unexpected response
                return 'no'  # Default to 'no' if there is an error
        except Exception as e:
            print(f"Error grading answer: {e}")
            return 'no'  # Default to 'no' if there is an error




### Reranker

class RelevanceScore(BaseModel):
    """Model representing the relevance score."""
    relevance_score: int = Field(
        description="The relevance score of the document with respect to the query, from 1 (least relevant) to 10 (most relevant)."
    )

class Document:
    """
    A simple Document class consisting of text content and optional metadata.
    """
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}
    
    def __repr__(self):
        return f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"

class Reranker:
    """
    Reranks a list of documents based on their relevance to a given query.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 1000,
        strict: bool = True,
    ):
        # Initialize the LLM
        self.llm = OpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            strict=strict
        )

        # Define the system prompt
        self.system_prompt = """You are an assistant that assigns a relevance score to a document based on its relevance to a query.
If the document contains keyword(s) or semantic meaning related to the user question, it is considered relevant.
The relevance score should be an integer from 1 (least relevant) to 10 (most relevant)."""

        # Create the FunctionTool instance inside the class
        self.tool = FunctionTool.from_defaults(
            fn=self.assign_relevance_score_function,
            name="assign_relevance_score"
        )

    @staticmethod
    def assign_relevance_score_function(relevance_score: int) -> RelevanceScore:
        """Assigns a relevance score to a document based on its relevance to a query."""
        return RelevanceScore(relevance_score=relevance_score)

    def rerank_documents(self, query: str, documents: List[Document], top_n: int = 3) -> List[dict]:
        """
        Reranks the provided documents based on their relevance to the query.

        Args:
            query (str): The user query.
            documents (List[Document]): A list of Document objects.

        Returns:
            List[dict]: The documents sorted by their relevance scores in descending order.
        """
        scored_documents = []

        for idx, doc in enumerate(documents):
            try:
                # Prepare the messages
                messages = [
                    ChatMessage(role="system", content=self.system_prompt),
                    ChatMessage(
                        role="user",
                        content=f"Query:\n{query}\n\nRetrieved document:\n{doc.page_content}\n\n"
                    ),
                ]

                # Call the LLM with the tool
                response = self.llm.predict_and_call(
                    [self.tool],
                    prompt=messages,
                )

                # Parse the response
                if response.additional_kwargs and 'function_call' in response.additional_kwargs:
                    function_call = response.additional_kwargs['function_call']
                    if function_call.get('name') == 'assign_relevance_score':
                        arguments = function_call.get('arguments')
                        # arguments is a JSON string
                        args = json.loads(arguments)
                        relevance_score_obj = RelevanceScore(**args)
                        relevance_score = relevance_score_obj.relevance_score
                    else:
                        relevance_score = 1  # Default to least relevant
                else:
                    relevance_score = 1  # Default to least relevant

                # Add the relevance score to the document
                doc_with_score = {
                    "content": doc,
                    "relevance_score": relevance_score
                }
                scored_documents.append(doc_with_score)
            except Exception as e:
                print(f"Error scoring document {idx}: {e}")
                doc_with_score = {
                    "content": doc,
                    "relevance_score": 1  # Default to least relevant
                }
                scored_documents.append(doc_with_score)

        # Sort the documents by relevance score in descending order
        sorted_documents = sorted(scored_documents, key=lambda x: x['relevance_score'], reverse=True)

        top_documents = sorted_documents[:top_n]

        return top_documents


### Rag Generator

class AnswerGenerator:
    """
    Generates an answer by combining the user's question with the retrieved context (documents).
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        # Initialize the LLM
        self.llm = OpenAI(model=model_name, temperature=temperature)
        
        # Define the prompt template using LlamaIndex's PromptTemplate
        self.prompt_template = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                        If you don't know the answer, just say that you don't know. 
                        Use three sentences maximum and keep the answer concise.
                        You will be given a step-back query to help retrieve relevant background information. 
                        You will also be given sub-queries that, when answered together, would provide a comprehensive response to the original query.
                        You must use the answers to both the step-back query and sub-queries to guide your final answer.
                        User query: {question} 
                        Step-back query: {step_back_query}
                        Sub-queries: {subqueries}
                        Context: {context} 
                        Answer:
                        """
        )
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Formats the retrieved documents into a single string to be used as context.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate_answer(self, question: str, docs: List[Document], step_back_query: str, subqueries: List[str]) -> Optional[str]:
        """
        Generates an answer to the question using the provided documents as context.
        
        Args:
            question (str): The user's question.
            docs (List[Document]): A list of Document objects to use as context.
            step_back_query (str): The step-back query.
            subqueries (List[str]): The list of sub-queries.
        
        Returns:
            str: The generated answer.
        """
        # Format the documents
        context = self.format_docs(docs)
        
        # Prepare the prompt using the PromptTemplate
        prompt = self.prompt_template.format(
            question=question,
            step_back_query=step_back_query,
            subqueries="\n".join(subqueries),
            context=context
        )
        
        # For chat models, convert the prompt to messages
        messages = self.prompt_template.format_messages(
            question=question,
            step_back_query=step_back_query,
            subqueries="\n".join(subqueries),
            context=context
        )
        
        # Since we're using a single prompt, the messages list will contain one message
        # You can adjust roles if needed; here we'll set the role as 'user'
        messages = [ChatMessage(role="user", content=prompt)]
        
        # Invoke the LLM to get the answer
        try:
            response = self.llm.chat(messages)
            answer = response.content.strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

class NoRetrievalGenerate:
    """
    Generates an answer to the user's question without using any retrieved documents.
    Uses only the LLM's built-in knowledge.
    """
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0):
        # Initialize the LLM
        self.llm = OpenAI(model=model_name, temperature=temperature)
        
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}")
            ]
        )
        
        # Create the chain
        #self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def generate_answer(self, question: str) -> str:
        """
        Generates an answer to the question using only the LLM's built-in knowledge.
        
        Args:
            question (str): The user's question.
        
        Returns:
            str: The generated answer.
        """
        try:
            # Format the messages
            messages = self.prompt_template.format_messages(question=question)
            
            # Invoke the LLM
            response = self.llm.chat(messages)
            answer = response.message.content.strip()
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

### Fusion Retrieval

def encode_text_and_get_split_nodes(text, chunk_size=1024):
    """
    Encodes text into a FAISS vector store using OpenAI embeddings.

    Args:
        text (str): The input text string.
        chunk_size (int): The desired size of each text chunk.

    Returns:
        tuple: A tuple containing the FAISS vector store and the list of nodes.
    """
    # Initialize the text splitter
    text_parser = SentenceSplitter(
        chunk_size=chunk_size,
    )

    # Split the text into chunks
    text_chunks = text_parser.split_text(text)

    # Create TextNode objects from chunks
    nodes = []
    for text_chunk in text_chunks:
        # Replace 't' with space in the text chunk
        cleaned_text = text_chunk.replace('t', ' ')
        node = TextNode(text=cleaned_text)
        nodes.append(node)

    # Initialize the OpenAI Embedding model
    embedding_model = OpenAIEmbedding()

    # Generate embeddings for each node
    embeddings = []
    for node in nodes:
        node_embedding = embedding_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
        embeddings.append(node_embedding)

    # Create the FAISS index
    embedding_dim = embedding_model.embedding_dimension
    index = faiss.IndexFlatL2(embedding_dim)

    # Create the FAISS vector store
    vector_store = FAISSVectorStore(
        faiss_index=index,
        embeddings=embedding_model,
    )

    # Add nodes to the vector store
    vector_store.add(nodes)

    return vector_store, nodes

def replace_t_with_space(documents: List[Document]) -> List[Document]:
    """
    Replaces '\t' characters with spaces in the document content.

    Args:
        documents (List[Document]): List of Document objects.

    Returns:
        List[Document]: Cleaned documents.
    """
    for doc in documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return documents

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """
    Create a BM25 index from the given documents.

    Args:
        documents (List[Document]): List of documents to index.

    Returns:
        BM25Okapi: An index that can be used for BM25 scoring.
    """
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

def fusion_retrieval(vectorstore, bm25, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
    """
    Perform fusion retrieval combining keyword-based (BM25) and vector-based search.

    Args:
        vectorstore (VectorStore): The vectorstore containing the documents.
        bm25 (BM25Okapi): Pre-computed BM25 index.
        query (str): The query string.
        k (int): The number of documents to retrieve.
        alpha (float): The weight for vector search scores (1-alpha will be the weight for BM25 scores).

    Returns:
        List[Document]: The top k documents based on the combined scores.
    """
    # Retrieve all documents
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)

    # BM25 scores
    bm25_scores = bm25.get_scores(query.split())

    # Vector search scores
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))
    vector_scores_dict = {doc.page_content: score for doc, score in vector_results}
    vector_scores = [vector_scores_dict.get(doc.page_content, 0) for doc in all_docs]

    # Normalize scores
    vector_scores = np.array(vector_scores)
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + 1e-8)
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-8)

    # Combine scores
    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    # Return top k documents
    return [all_docs[i] for i in sorted_indices[:k]]

class FusionRetrievalRAG:
    def __init__(self, file_path: str, chunk_size: int = 200, chunk_overlap: int = 200):
        """
        Initializes the FusionRetrievalRAG class by setting up the vector store and BM25 index.

        Args:
            file_path (str): Path to the input file (PDF, DOCX, Markdown).
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive chunks.
        """
        # Process the file and perform coreference resolution
        resolved_text = process_file(file_path)

        # Encode the text and get split documents
        self.vectorstore, self.cleaned_texts = encode_text_and_get_split_documents(
            resolved_text, chunk_size, chunk_overlap
        )

        # Create BM25 index
        self.bm25 = create_bm25_index(self.cleaned_texts)

    def run(self, query: str, k: int = 5, alpha: float = 0.5):
        """
        Executes the fusion retrieval for the given query.

        Args:
            query (str): The search query.
            k (int): The number of documents to retrieve.
            alpha (float): The weight of vector search vs. BM25 search.

        Returns:
            List[str]: The content of the top k retrieved documents.
        """
        top_docs = fusion_retrieval(self.vectorstore, self.bm25, query, k, alpha)
        docs_content = [doc.page_content for doc in top_docs]
        return docs_content


### Construct Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

def retrieve(state, file_path: str):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state
        file_path (str): Path to the input file for retrieval
    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = FusionRetrievalRAG(file_path).run(query=question, k=5, alpha=0.7)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    step_back_query = QueryTransformer().generate_stepback_query({"question": question})
    subqueries = QueryTransformer().decompose_question({"question": question})

    # RAG generation
    generation = AnswerGenerator().generate_answer({
        "context": documents, 
        "question": question,
        "step_back_query": step_back_query,
        "subqueries": subqueries
        })
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state,
                    score_cutoff: int = 4):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state
        score_cutoff: The relevance threshold with which to retain a document in the context

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    top_documents = Reranker().rerank_documents(
        {"query": question, "documents": documents}
        )
    for idx, doc in enumerate(top_documents):
        if doc['relevance_score'] >= score_cutoff:
            print(f"---DOCUMENT GRADE: {doc['relevance_score']}\n DOCUMENT {idx} RELEVANT---""")
            filtered_docs.append(doc['content'])
        else:
            print(f"---DOCUMENT GRADE: {doc['relevance_score']}\n DOCUMENT {idx} NOT RELEVANT---""")
            continue
    return filtered_docs


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = QueryTransformer().rewrite_question({"question": question})
    return {"documents": documents, "question": better_question}


def noretreivalgenerate(state):
    """
    Ask LLM re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Answer to user query
    """
    print("---QUERY LLM DIRECTLY---")
    question = state['question']
    better_question = QueryTransformer().rewrite_question({"question": question})
    return NoRetrievalGenerate().generate_answer({"question": better_question})



### Edges ###


def route_question(state, knowledge_base_description: str):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state
        knowledge_base_description (str): Description of the knowledge base

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = QueryAnalyzer(knowledge_base_description).analyze_query(question)
    if source == "no-retrieval":
        print("---ROUTE QUESTION TO LLM BUILT IN KNOWLEDGE---")
        return "no-retrieval"
    elif source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    grade = HallucinationGrader().grade_hallucination(
        {"documents": documents, 
        "generation": generation}
    )

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        grade = AnswerGrader().grade_answer({"question": question, "generation": generation})
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"




class LangGraphApp:
    """
    Class to encapsulate the LangGraph workflow for RAG capabilities.
    """

    def __init__(self):
        # Initialize the graph
        self.workflow = StateGraph(GraphState)

        # Define the nodes
        self.workflow.add_node("no-retrieval", noretreivalgenerate)  # Query LLM directly
        self.workflow.add_node("retrieve", retrieve)  # Retrieve documents
        self.workflow.add_node("grade_documents", grade_documents)  # Grade documents
        self.workflow.add_node("generate", generate)  # Generate answer
        self.workflow.add_node("transform_query", transform_query)  # Transform query

        # Build graph
        self.workflow.add_conditional_edges(
            START,
            route_question,
            {
                "no-retrieval": "no-retrieval",
                "vectorstore": "retrieve",
            },
        )
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("transform_query", "retrieve")
        self.workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

    def compile_app(self):
        """
        Compile the workflow into a callable app.

        Returns:
            Callable: The compiled app.
        """
        return self.workflow.compile()


# Function to initialize and compile the LangGraph app
def create_langgraph_app(knowledge_base_description: str, file_path: str):
    """
    Create and return a LangGraph application for RAG.

    Args:
        knowledge_base_description (str): Description of the knowledge base
        file_path (str): Path to the input file for retrieval

    Returns:
        Callable: The compiled LangGraph app.
    """
    langgraph_app = LangGraphApp(knowledge_base_description, file_path)
    return langgraph_app.compile_app()


