
from llama_cpp import Llama
from datetime import datetime
from langchain.vectorstores import FAISS 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader


cfg = {
    'db_dir' : './Desktop/Python/FAISS/ManU',
    'file_dir' : './Desktop/Python/FAISS/DATA',
    #'model_path' : './llama.cpp/models/mistral-7b-openorca.Q4_0.gguf',
    'model_path' : './Documents/GitHub/llama.cpp/models/mistral-7b-openorca.Q4_0.gguf',
    'vector_query' : 'What is the commercial deal with Adidas?',
    'sum_prompt_1' : 'You are an equity research analyst. Read the following article and summarize it as a list of facts. Have only one fact per line. Only include facts that are directly mentioned in the article. Article: ',
    'sum_prompt_2' : 'You are an equity research analyst. Summarize the provided article. Only include facts that are directly mentioned in the article. Prioritize financial amounts, agreement term lengths (years), and operating terms. Do not include information not about the agreement with Adidas. Article: '
}

# --- FAIS Functions

def input_data():
    #Define File Locations
    loader = DirectoryLoader(cfg['file_dir'], glob='*.pdf', loader_cls=PyPDFLoader)
    pages = loader.load_and_split()

    #Define Splitting
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    return docs

def embed_save(data):
    # Define Embedding and Saving
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(data, embeddings)
    db.save_local(cfg['db_dir'])
    print("Save complete!")

def load_data():
    #Loading Data
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    loaded_data = FAISS.load_local(cfg['db_dir'], embeddings)
    return loaded_data

def v_query(v_db):
    #Similarity Search
    
    query = cfg['vector_query']
    docs_and_scores = v_db.similarity_search_with_score(query, 3)

    for doc, score in docs_and_scores:
        print('-'* 60)
        #print(f'Source Text: {doc.page_content}')
        #print(f'Page Number: {doc.metadata["page"]}')
        print(f'Document Name: {doc.metadata}')
        print(f'Score : {score}\n')

        #result_string = str(doc.page_content) + '[[Page]]' + str(doc.metadata["page"])
        #v_result.append(result_string)
        #llm_sum(result_string)

    return docs_and_scores

# --- LLM Functions

def llm_sum(v_data):
   
    llm = Llama(model_path=cfg['model_path'], n_ctx=4096, n_gpu_layers = 30, verbose= False) 

    prompt = cfg['sum_prompt_1']
    article = str(v_data)
    query = prompt + article
    single_turn_prompt = f"GPT4 Correct User: {query}<|end_of_turn|>GPT4 Correct Assistant:"
    
    output = llm(single_turn_prompt, max_tokens=4096)
    text_content = output['choices'][0]['text']
    llm_r1.append(str(text_content))


def llm_write(v_data):
   
    llm = Llama(model_path=cfg['model_path'], n_ctx=4096, n_gpu_layers = 30, verbose= False) 

    prompt = cfg['sum_prompt_2']
    article = str(v_data)
    #print (article)

    query = prompt + article
    single_turn_prompt = f"GPT4 Correct User: {query}<|end_of_turn|>GPT4 Correct Assistant:"
    
    output = llm(single_turn_prompt, max_tokens=4096)
    text_content = output['choices'][0]['text']

    print('-'*80)
    print(text_content)
    print('-'*80)
    print(output)

    return str(output)

def output_backup():

    with open('./Desktop/Python/FAISS/sum_result_0.txt', 'w') as file:
        file.write('\n'.join(v_result))

    with open('./Desktop/Python/FAISS/sum_result_1.txt', 'w') as file:
        file.write('\n'.join(llm_r1))

    with open('./Desktop/Python/FAISS/sum_result_2.txt', 'w') as file:
        file.write(llm_r2)

#---------------------------------------------------------------------------
# MAIN
#---------------------------------------------------------------------------

# --------- For Timing
print('-'*110)
start = datetime.now()
print(start)


# -------------------------------------- Primary Operations

#documents = input_data()    #Generates Embeddings - Only need to run once
#embed_save(documents)       #Generates Embeddings - Only need to run once

v_result = []
llm_r1 = []
vector_db = load_data()
test_data = v_query(vector_db)

#print(llm_r1)
#llm_r2 = llm_write(llm_r1)
llm_r2 = llm_write(test_data)
#output_backup()


# --------- For Timing
finish = datetime.now()
time = finish - start
print(time)
