from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler,
)  # for streaming resposne
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.faiss import FAISS


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Alt way to load LLM
# def load_llm(model_file):
#     llm = CTransformers(
#         model=model_file,
#         model_type="llama",
#         max_new_tokens=1024,
#         temperature=0.01
#     )
#     return llm

def load_llm(model_file):
    llm = LlamaCpp(
        model_path=model_file,
        # n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # seed=1337, # Uncomment to set a specific seed
        n_ctx=1024, # Uncomment to increase the context window
        verbose=False
    )
    return llm


# Tao prompt template
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=['context',"question"])
    return prompt


# Tao simple chain
def create_simple_chain(prompt, llm):
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=1024),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}
    )
    return llm_chain

# Read tu VectorDB
def read_vectors_db(model_file,vector_db_path):
    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file=model_file)
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db


