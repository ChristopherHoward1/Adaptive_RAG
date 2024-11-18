
from typing import Literal, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain import hub


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "no-retrieval"] = Field(
        ...,
        description="Given a user question choose to route it to a vectorstore or to the LLMs built in context.",
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
        for idx, question in enumerate(subqueries):
            print(f"Subqueries\n Question {idx}: {question}")
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
    relevance_score: int = Field(gt=0, le=5, description="The relevance score of a document to a query.")


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
                                The relevance score should be an integer from 1 (least relevant) to 5 (most relevant)."""
        
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




### Hallucination Grader

# class GradeHallucinations(BaseModel):
#     """Binary score for hallucination present in generation answer."""

#     binary_score: str = Field(
#         description="Answer is grounded in the facts, 'yes' or 'no'"
#     )

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
