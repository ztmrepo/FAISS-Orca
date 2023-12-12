
import pandas as pd
from llama_cpp import Llama
from datetime import datetime
from langchain.vectorstores import FAISS 
from langchain.embeddings import SentenceTransformerEmbeddings


cfg = {
    'db_dir' : './Desktop/Python/Fedramp/FAISS/',
    'file_dir' : './Desktop/Python/Fedramp/',
    'file_path' : './Desktop/Python/Fedramp/fedramp.csv',
    #'model_path' : './llama.cpp/models/mistral-7b-openorca.Q4_0.gguf',
    'model_path' : './Documents/GitHub/llama.cpp/models/mistral-7b-openorca.Q4_0.gguf',
    'vector_query' : 'What companies offer AI products?',
    'sum_prompt_1' : 'You are an equity research analyst. Determine, what AI features are offered by what companies. Only include facts that are directly mentioned in the article. Article: '
}

# --- FAIS Functions

def input_csv():
    passage_data = pd.read_csv(cfg['file_path'])
    passage_data['Service Offering'] = passage_data['Service Offering'].astype(str)
    return passage_data

def embed_save(data):
    # Define Embedding and Saving
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    metadatas = []
    for index, row in data.iterrows(): #Needs index even though not referneced
        doc_meta = {
            "Provider": (row['Provider']),
            "Service Offering": (row['Service Offering']),
            "Service Model": (row['Service Model']),
            "Impact Level": (row['Impact Level']),
            "Status": (row['Status']),
        }
        metadatas.append(doc_meta)

    db = FAISS.from_texts(data['Service Offering'].tolist(), embeddings, metadatas)
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
    docs_and_scores = v_db.similarity_search_with_score(query, 5)

    for doc, score in docs_and_scores:
        print('-'* 60)
        #print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata}')
        print(f'Score : {score}\n')

    return docs_and_scores

# --- LLM Functions


def llm_write(v_data):
   
    llm = Llama(model_path=cfg['model_path'], n_ctx=4096, n_gpu_layers = 35, verbose= False) 

    query = cfg['sum_prompt_1'] + str(v_data)
    single_turn_prompt = f"GPT4 Correct User: {query}<|end_of_turn|>GPT4 Correct Assistant:"
    
    output = llm(single_turn_prompt, max_tokens=4096)
    text_content = output['choices'][0]['text']

    print('-'*80)
    print(text_content)
    print('-'*80)
    print(output)

    return str(output)

def output_backup():

    v_floc = cfg['db_dir'] + 'sum_result_0.txt'
    l1_floc = cfg['db_dir'] + 'sum_result_1.txt'

    with open(v_floc, 'w') as file:
        file.write(str(test_data))

    with open(l1_floc, 'w') as file:
        file.write(llm_r1)

#---------------------------------------------------------------------------
# MAIN
#---------------------------------------------------------------------------

# --------- For Timing
print('-'*110)
start = datetime.now()
print(start)


# -------------------------------------- Primary Operations

documents = input_csv()    #Generates Embeddings - Only need to run once
embed_save(documents)       #Generates Embeddings - Only need to run once

v_result = []
vector_db = load_data()
test_data = v_query(vector_db)
llm_r1 = llm_write(test_data)
output_backup()


# --------- For Timing
finish = datetime.now()
time = finish - start
print(time)
