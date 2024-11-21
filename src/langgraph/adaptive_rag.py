import numpy as np

from typing import Literal, List
from typing_extensions import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain import hub
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from typing import List
from rank_bm25 import BM25Okapi
from pprint import pprint
from langgraph.graph import END, StateGraph, START


from helper_fns import process_file

# Variables


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "no-retrieval"] = Field(
        ...,
        description="Given a user query choose to route it to a vectorstore or to the LLMs built in context.",
    )

### Query Analyzer

class QueryAnalyzer:
    """
    Classifies the user query to determine if it requires retrieval from the knowledge base
    or can be answered directly by the LLM.
    """
    def __init__(
        self,
        knowledge_base_description: str,
        model_name: str = "gpt-4o",
        temperature: float = 0
    ):
        # Initialize the LLM with function calling capabilities
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens = 4000)
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        
        # Define the system prompt with the user-provided knowledge base description
        self.system_prompt = f"""You are an expert at routing a user question to a vectorstore or to an LLM.
                                The vectorstore contains documents related to: {knowledge_base_description}
                                Use the vectorstore for questions on these topics. Otherwise, use no-retrieval."""
        
        # Create the prompt template
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )

        self.question_router = self.route_prompt | self.structured_llm_router

    def analyze_query(self, question: str) -> str:
        """
        Determines the query's complexity and relevance.
        Returns 'vectorstore' or 'no-retrieval' based on the analysis.
        """
        # Format the prompt with the user's question
        formatted_prompt = self.route_prompt.format(question=question)
        
        # Get the structured output from the LLM
        response = self.structured_llm_router(formatted_prompt)
        
        # Parse the response into the RouteQuery model
        route_query = RouteQuery.parse_raw(response.content)
        
        return route_query.datasource
    




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
    Generates a step-back query to get broad context
    Decompose original query into simpler subqueries
    """
    def __init__(self,
                model_name: str = "gpt-4o"
                ):
        # Initialize the LLM
        self.rewrite_llm = ChatOpenAI(model=model_name, temperature=0)
        self.stepback_llm = ChatOpenAI(model=model_name, temperature=0)
        self.subquery_llm = ChatOpenAI(model=model_name, temperature=0.3)


        # Define the system prompts
        self.rewrite_system_prompt = (
            """You are an AI assistant that converts an input question to a better version 
            that is optimized for vectorstore retrieval. Look at the input and try to reason 
            about the underlying semantic intent/meaning."""
        )
        self.stepback_system_prompt = (
            """You are an AI assistant that takes input questions and generates more broad, 
            general queries to improve context retreival in the RAG system.
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

        # Structured outputs for each query transform
        self.structured_llm_rewrite = self.llm.with_structured_output(RewrittenQuery)
        self.structured_llm_stepback = self.llm.with_structured_output(StepBackQuery)
        self.structured_llm_subqueries = self.llm.with_structured_output(SubQueries)


        # Create the prompt re-writer template
        self.rewrite_chain = ChatPromptTemplate.from_messages(
            [
                ("system", self.rewrite_system_prompt),
                (
                    "human",
                    "Here is the initial question:\n\n{question}\n\nFormulate an improved question.",
                ),
            ]
        )
        
        # Create the prompt step-back template
        self.stepback_chain = ChatPromptTemplate.from_messages(
            [
                ("system", self.stepback_system_prompt),
                (
                    "human",
                    "Here is the initial question:\n\n{question}\n\nFormulate a more general question that captures the semantic meaning of the original.",
                ),
            ]
        )

        # Create the query decomp prompt template
        self.decomp_chain = ChatPromptTemplate.from_messages(
            [
                ("system", self.subquery_system_prompt),
                (
                    "human",
                    "Here is the initial question:\n\n{question}\n\nFormulate sub-questions that help answer the original question.",
                ),
            ]
        )


        # Create the pipeline: prompt -> llm -> output parser
        self.rewrite_pipeline = self.rewrite_chain | self.structured_llm_rewrite
        self.stepback_pipeline = self.stepback_chain | self.structured_llm_stepback
        self.decomp_pipeline = self.decomp_chain | self.structured_llm_subqueries

    def rewrite_question(self, question: str) -> str:
        """Rewrites the input question to improve vectorstore retrieval performance."""
        rewritten_query_obj = self.rewrite_chain.invoke({"question": question})
        return rewritten_query_obj.rewritten_query

    def generate_stepback_query(self, question: str) -> str:
        """Generates a more general step-back query based on the input question."""
        stepback_query_obj = self.stepback_chain.invoke({"question": question})
        return stepback_query_obj.stepback_query

    def decompose_question(self, question: str) -> List[str]:
        """Decomposes the input question into simpler sub-questions."""
        subqueries = self.decomp_chain.invoke({"question": question}).subqueries
        print("Decomposed Query:")
        for idx, question in enumerate(subqueries):
            print(f"Sub-query {idx}: {question}")
        return subqueries



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
        model_name: str = "gpt-4",
        temperature: float = 0,
        max_tokens: int = 1000
    ):
        # Initialize the LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)

        # Define the output parser
        #self.output_parser = PydanticOutputParser(pydantic_object=HallucinationGradingResult)
        self.structured_llm_grader = self.llm.with_structured_output(HallucinationGradingResult)

        # Get the format instructions for the output parser
        #format_instructions = self.output_parser.get_format_instructions()

        # Define the system prompt
        self.system_prompt = """You are an evaluator determining if an LLM-generated response is based on or supported by a provided set of retrieved facts.
                                Provide a binary score: 'yes' or 'no'.
                                'Yes' means the answer is grounded in and supported by the facts."""

        # Create the prompt template
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Set of facts:\n\n{documents}\n\nLLM generation:\n\n{generation}\n\n{format_instructions}",
                ),
            ]
        )

        # Create the LLMChain
        self.chain = self.hallucination_prompt | self.structured_llm_grader

    def grade_hallucination(self, documents: str, generation: str) -> str:
        """
        Grades the LLM-generated answer to determine if it is grounded in the provided documents.
        Returns 'yes' or 'no' based on the grading.
        """
        try:
            # Use chain.invoke to get the structured LLM response
            grading_result = self.chain.invoke({"documents": documents, "generation": generation})
            return grading_result.score.lower()
        except Exception as e:
            print(f"Error grading hallucination: {e}")
            return 'no'  # Default to 'no' if there is an error

### Answer Grader

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
class AnswerGrader:
    """
    Grades an LLM-generated answer to determine if it adequately addresses the user's question.
    """
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0
    ):
        
        
        # Initialize the LLM with function calling capabilities
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, max_tokens = 4000)
        self.structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        self.system_prompt = """You are a grader assessing whether an answer addresses / resolves a question \n 
                    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                    ("system", self.system_prompt),
                    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        self.hallucination_grader = self.answer_prompt | self.structured_llm_grader

    def grade_answer(self, question: str, generation: str) -> str:
        """
        Grades the answer generated by the LLM to determine if it resolves the user's question.
        Returns 'yes' or 'no' based on the grading.
        """
        # Format the prompt with the question and the generated answer
        #formatted_prompt = self.answer_prompt.format(question=question, generation=generation)
        
        # Get the structured output from the LLM
        grading_result = self.hallucination_grader.invoke({"question": question,
                                                            "generation": generation}).binary_score
        
        return grading_result.lower()



### Reranker

class RelevanceScore(BaseModel):
    relevance_score: int = Field(gt=0, le=10, description="The relevance score of a document to a query.")


class Reranker:
    """
    Reranks a list of documents based on their relevance to a given query.
    """
    def __init__(
        self,
        model_name: str = "gpt-4",
        temperature: float = 0,
        max_tokens: int = 1000
    ):
        # Initialize the LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        
        # Define the output parser
        self.structured_llm_reranker = self.llm.with_structured_output(RelevanceScore)
        
        
        # Define the system prompt
        self.system_prompt = """You are an assistant that assigns a relevance score to a document based on its relevance to a query.\n
                                If the document contains keyword(s) or semantic meaning related to the user question, it is considered relevant. \n
                                The relevance score should be an integer from 1 (least relevant) to 10 (most relevant)."""
        
        # Create the prompt template
        self.rerank_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Query:\n{query}\n\n Retrieved document:\n{document}\n\n"
                ),
            ]
        )
        
        # Create the LLMChain
        self.retrieval_grader = self.rerank_prompt | self.structured_llm_reranker

    def rerank_documents(self, query: str, documents: List[Document], top_n: int = 5) -> List[dict]:
        """
        Reranks the provided documents based on their relevance to the query.
        
        Args:
            query (str): The user query.
            documents (List[dict]): A list of documents, each with 'document_id' and 'content'.
        
        Returns:
            List[dict]: The documents sorted by their relevance scores in descending order.
        """
        scored_documents = []
        
        for idx, doc in enumerate(documents):
            try:
                # Prepare the input for the chain
                chain_input = {
                    "query": query,
                    "document": doc
                }
                # Invoke the chain to get the relevance score
                scoring_result = self.chain.invoke(chain_input)
                relevance_score = scoring_result.relevance_score
                
                # Add the relevance score to the document
                doc_with_score = {
                    "content": doc,
                    "relevance_score": relevance_score
                }
                scored_documents.append(doc_with_score)
            except Exception as e:
                print(f"Error scoring document doc {idx}: {e}")
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
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0):
        # Initialize the LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Pull the prompt from the hub
        self.prompt = hub.pull("rlm/rag-prompt")
        
        # Create the output parser
        self.output_parser = StrOutputParser()
        
        # Create the LLMChain
        self.chain = self.prompt | self.llm | self.output_parser
    
    def format_docs(self, docs):
        """
        Formats the retrieved documents into a single string to be used as context.
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def generate_answer(self, question: str, docs):
        """
        Generates an answer to the question using the provided documents as context.
        
        Args:
            question (str): The user's question.
            docs (List[Document]): A list of Document objects to use as context.
        
        Returns:
            str: The generated answer.
        """
        # Format the documents
        context = self.format_docs(docs)
        
        # Prepare the input for the chain
        chain_input = {
            "context": context,
            "question": question
        }
        
        # Invoke the chain to get the answer
        try:
            answer = self.chain.invoke(chain_input)
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

class NoRetrievalGenerate:
    """
    Generates an answer to the user's question without using any retrieved documents.
    Uses only the LLM's built-in knowledge.
    """
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0):
        # Initialize the LLM
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                ("human", "{question}")
            ]
        )
        
        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()
    
    def generate_answer(self, question: str) -> str:
        """
        Generates an answer to the question using only the LLM's built-in knowledge.
        
        Args:
            question (str): The user's question.
        
        Returns:
            str: The generated answer.
        """
        try:
            answer = self.chain.invoke({"question": question})
            return answer
        except Exception as e:
            print(f"Error generating answer: {e}")
            return None

### Fusion Retrieval

def encode_text_and_get_split_documents(text, chunk_size=200, chunk_overlap=200):
    """
    Encodes text into a vector store using OpenAI embeddings.

    Args:
        text (str): The input text string.
        chunk_size (int): The desired size of each text chunk.
        chunk_overlap (int): The amount of overlap between consecutive chunks.

    Returns:
        tuple: A tuple containing the FAISS vector store and the list of cleaned documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    documents = text_splitter.create_documents([text])

    cleaned_texts = replace_t_with_space(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore, cleaned_texts

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

    # RAG generation
    generation = AnswerGenerator.generate_answer({"context": documents, "question": question})
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
    return NoRetrievalGenerate().generate_answer({"question": question})



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
        {"documents": documents, "generation": generation}
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


