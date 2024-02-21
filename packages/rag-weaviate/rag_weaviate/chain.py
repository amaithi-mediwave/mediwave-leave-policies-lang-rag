

from langchain_community.vectorstores import Weaviate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.cache import RedisSemanticCache
from langchain.globals import set_llm_cache

from dotenv import load_dotenv

load_dotenv()

import os
import weaviate
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#---------------------- ENV ---------------------------------------


WEAVIATE_CLIENT_URL = os.getenv('WEAVIATE_CLIENT_URL')
WEAVIATE_COLLECTION_NAME = os.getenv('WEAVIATE_COLLECTION_NAME')
WEAVIATE_COLLECTION_PROPERTY = os.getenv('WEAVIATE_COLLECTION_PROPERTY')
REDIS_URL_SEMANTIC_CACHE = os.getenv('REDIS_URL_SEMANTIC_CACHE')
WEAVIATE_RETRIEVER_SEARCH_TOP_K = os.getenv('WEAVIATE_RETRIEVER_SEARCH_TOP_K')
OLLAMA_SERVER_URL = os.getenv('OLLAMA_SERVER_URL')
CHAT_OLLAMA_MODEL_NAME = os.getenv('CHAT_OLLAMA_MODEL_NAME')
CHAT_OLLAMA_MODEL_TEMPERATURE = os.getenv('CHAT_OLLAMA_MODEL_TEMPERATURE')
REDIS_URL_CHAT_MEMORY_HISTORY = os.getenv('REDIS_URL_CHAT_MEMORY_HISTORY')

#------------------------------ REDIS SEMANTIC CACHE -----------------------

set_llm_cache(RedisSemanticCache(
    embedding=GPT4AllEmbeddings(),
    redis_url=REDIS_URL_SEMANTIC_CACHE
))


#------------------------------ WEAVIATE VECTOR STORE -----------------------

client = weaviate.Client(url=WEAVIATE_CLIENT_URL)

vectorstore = Weaviate(client, 
                       WEAVIATE_COLLECTION_NAME, 
                       WEAVIATE_COLLECTION_PROPERTY)

retriever = vectorstore.as_retriever(search_kwargs={"k": int(WEAVIATE_RETRIEVER_SEARCH_TOP_K)})


#------------------------------ LLM - Llama2 -----------------------


chat_model = ChatOllama(
    base_url=OLLAMA_SERVER_URL,
    model=CHAT_OLLAMA_MODEL_NAME, 
    temperature=float(CHAT_OLLAMA_MODEL_TEMPERATURE))


question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You're an Friendly AI assistant, your name is Claro, you can make normal conversations in a friendly manner, and also provide Answer to the human question based on the given contents,
            
            if the question is any related or referenced the chat history entities make use of that to obtain the context to answer the the given question effectively. if the given question is a followup question, then find the context and then answer to it. don't use chat history if the question is standalone and make sure to answer the given question alone. if no chat history is provided then continue with the question alone. 
            following is the chat summary : \n {chat_history} and contents : \n {context} answer the following human question using either chat summary or context or based on both, if the question has nothing related to the chat summary or  context then answer on your own don't use the context. make sure it sounds like human and official assistant. don't use any words provided in the prompt. """
        
        ),
        ("human", "\n{question}")

    ]
)

#------------------------------ CHAIN -----------------------

document_chain = create_stuff_documents_chain(chat_model, question_answering_prompt)   

conversational_retrieval_chain = RunnablePassthrough.assign(
    
    context = (lambda x: x["question"]) | retriever).assign(
    answer=document_chain,) | itemgetter("answer")


#------------------------------ MESSAGE HISTORY CHAIN -----------------------
chain = RunnableWithMessageHistory(
    conversational_retrieval_chain,

    lambda session_id: RedisChatMessageHistory(
        session_id, 
        url=REDIS_URL_CHAT_MEMORY_HISTORY
    ),
    
    input_messages_key="question",
    history_messages_key="chat_history",
    #  history_messages_key="history",
    # output_messages_key="answer"
)



class ChatBot(BaseModel):
    question: str
    


# from langchain.globals import set_debug
# set_debug(True)
# from langchain.globals import set_verbose
# set_verbose(True)


chain = chain.with_types(input_type=ChatBot)
