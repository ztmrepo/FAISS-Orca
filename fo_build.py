
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from llama_cpp import Llama
from datetime import datetime


cfg = {
    'db_dir' : './Desktop/FAISS-PC/ManU',
    'file_dir' : './Desktop/FAISS-PC/DATA',
    'model_path' : './Documents/GitHub/llama.cpp/models/mistral-7b-openorca.Q4_0.gguf',
    'vector_query' : 'What is the commercial deal with Adidas?',
    'sum_prompt_1' : 'You are an equity research analyst. Read the following article and summarize it as a list of facts. Have only one fact per line. Only include facts that are directly mentioned in the article. Ignore information not related to Adidas. Article: ',
    'sum_prompt_2' : 'You are an equity research analyst. Take this article which is is structured as a list of facts and turn it into a written summary of the deal with Adidas. Only include facts that are directly mentioned in the article. Only include information relevant to the Adidas relationship. Prioritize financial amounts, agreement term lengths (years), and operating terms. Article: '
        
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

        result_string = str(doc.page_content) + '[[Page]]' + str(doc.metadata["page"])
        llm_sum(result_string)

    return docs_and_scores

# --- LLM Functions

def llm_sum(v_data):
   
    llm = Llama(model_path=cfg['model_path'], n_ctx=4096, n_gpu_layers = 30, verbose= False) 

    prompt = cfg['sum_prompt_1']
    article = str(v_data)
    query = prompt + article
    
    output = llm(query, max_tokens=4096)
    text_content = output['choices'][0]['text']
    llm_r1.append(str(text_content))


def llm_write(v_data):
   
    llm = Llama(model_path=cfg['model_path'], n_ctx=4096, n_gpu_layers = 30) 

    prompt = cfg['sum_prompt_2']
    article = str(v_data)
    query = prompt + article
    
    output = llm(query, max_tokens=4096)
    text_content = output['choices'][0]['text']

    print('-'*80)
    print(text_content)
    print('-'*80)
    print(output)

    return text_content

def output_backup():
    with open('./Desktop/FAISS-PC/sum_result_1.txt', 'w') as file:
        file.write('\n'.join(llm_r1))

    with open('./Desktop/FAISS-PC/sum_result_2.txt', 'w') as file:
        file.write(llm_r2)

#---------------------------------------------------------------------------
# MAIN
#---------------------------------------------------------------------------



# -------------------------------------- Primary Operations

documents = input_data()    #Generates Embeddings - Only need to run once
embed_save(documents)       #Generates Embeddings - Only need to run once

# --------- For Timing
print('-'*110)
start = datetime.now()
print(start)


llm_r1 = []
vector_db = load_data()
v_query(vector_db)
llm_r2 = llm_write(llm_r1)
#output_backup()


# --------- For Timing
finish = datetime.now()
time = finish - start
print(time)
