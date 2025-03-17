import grpc
from concurrent import futures
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

import QueryRewriterService_pb2
import QueryRewriterService_pb2_grpc

# Load an open-source LLM (Mistral-7B or t5-base for lightweight usage)
hf_pipeline = pipeline("text2text-generation", model="facebook/opt-iml-max-1.3b", max_new_tokens=50)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the LangChain prompt
QUERY_EXPANSION_PROMPT = PromptTemplate.from_template(
    """You are an expert in information retrieval.
Your goal is to improve search results by generating alternative phrasings of the query. \
Please provide {num_variants} different ways to ask this same question. The variations should: \
- Retain the original meaning \
- Use different wording \
- Sound natural to the user \
Write each query on a new line. \
YOU MUST PROVIDE EXACTLY {num_variants} QUERIES. You will be fined it doing otherwise. \
EXAMPLE: \
Original query: How to feed dogs?
Response:
Dogs feeding process
Ways to feed dogs
Dogs food
END OF EXAMPLE
Original query: {question}
Response:
"""
)

MULTIQUERY_GEN = QUERY_EXPANSION_PROMPT | llm | StrOutputParser() | (lambda x: x.split("\n"))

# gRPC Service Implementation
class QueryRewriterService(QueryRewriterService_pb2_grpc.QueryRewriterServicer):
    def rephrase(self, request, context):
        print(f"Received query: {request.query} with {request.num_variants} variants")

        expanded_queries = MULTIQUERY_GEN.invoke({"question": request.query, "num_variants": request.num_variants})

        # Ensure the correct number of responses
        expanded_queries = expanded_queries[: request.num_variants] if len(expanded_queries) >= request.num_variants else expanded_queries + [""] * (request.num_variants - len(expanded_queries))

        return QueryRewriterService_pb2.QueryResponse(variants=expanded_queries)

# Start gRPC server
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    QueryRewriterService_pb2_grpc.add_QueryRewriterServicer_to_server(QueryRewriterService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Query Rewriter gRPC Server started on port 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
