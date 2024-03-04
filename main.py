# Chay thu chain

from rag_chain import create_prompt, create_qa_chain, load_llm, read_vectors_db

# Cau hinh
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời
    {context}<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    |im_start|>assistant"""

db = read_vectors_db(model_file,vector_db_path)
prompt = create_prompt(template)
llm = load_llm(model_file)

# llm_chain = create_simple_chain(prompt, llm)
# question = "Tam giác có mấy cạnh?"
# response = llm_chain.invoke({"question":question})
# print(response)

llm_chain = create_qa_chain(prompt, llm, db)
question = "How to choose an AWS Region?"
response = llm_chain.invoke({"query": question})
print(response)
