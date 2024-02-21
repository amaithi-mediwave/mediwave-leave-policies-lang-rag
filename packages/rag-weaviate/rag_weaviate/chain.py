import os
import os
from dotenv import load_dotenv

load_dotenv()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from langchain_community.vectorstores import Weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.chat_models import ChatOllama

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

import weaviate
from langchain.globals import set_llm_cache
from langchain.cache import RedisCache
import redis
from retry import retry



WEAVIATE_CLIENT_URL = os.getenv('WEAVIATE_CLIENT_URL')
WEAVIATE_COLLECTION_NAME = os.getenv('WEAVIATE_COLLECTION_NAME')
WEAVIATE_COLLECTION_PROPERTY = os.getenv('WEAVIATE_COLLECTION_PROPERTY')
REDIS_URL_SEMANTIC_CACHE = os.getenv('REDIS_URL_SEMANTIC_CACHE')
WEAVIATE_RETRIEVER_SEARCH_TOP_K = os.getenv('WEAVIATE_RETRIEVER_SEARCH_TOP_K')
OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL')
CHAT_OLLAMA_MODEL_NAME = os.getenv('CHAT_OLLAMA_MODEL_NAME')


redis_client = redis.Redis.from_url(REDIS_URL_SEMANTIC_CACHE)
set_llm_cache(RedisCache(redis_client))


client = weaviate.Client(
  url=WEAVIATE_CLIENT_URL,
)

vectorstore = Weaviate(client, 
                       WEAVIATE_COLLECTION_NAME,
                       WEAVIATE_COLLECTION_PROPERTY 
                       )

retriever = vectorstore.as_retriever(search_kwargs={"k": int(WEAVIATE_RETRIEVER_SEARCH_TOP_K)})




# RAG prompt
template = """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, and also provide Answer the question based on the following context make sure it sounds like human and official assistant. if you're not sure about the question with related to context then ask the user what you want and if needed you can give the reference document details or links:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)




# RAG
model = ChatOllama(
    base_url=OLLAMA_SERVER_URL,
    model=CHAT_OLLAMA_MODEL_NAME)

chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)



# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)

