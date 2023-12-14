
import faiss
import numpy as np
import pandas as pd
from llama_cpp import Llama
from datetime import datetime
from sentence_transformers import SentenceTransformer


cfg = {
    'file_path' : './Desktop/Python/Stratechery/test2.csv',
    'embed_path' : './Desktop/Python/NoChain/stratechery.bin',
    'model_path' : './Documents/GitHub/llama.cpp/models/mistral-7b-openorca.Q4_0.gguf',
    #'model_path' : './Documents/GitHub/llama.cpp/models/starling-lm-7b-alpha.Q4_0.gguf',
   }

def create_embeddings():
    #Step 1 - Define Data
    data = pd.read_csv(cfg['file_path'])

    #Step 2 - Create Vectors from Text
    text = data['Body'].astype(str)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = encoder.encode(text)

    #Step 3 - Build FAISS Index
    vector_dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    ### Read / Write Indexes ###
    faiss.write_index(index, cfg['embed_path'])

def vector_query():
        
    data = pd.read_csv(cfg['file_path'])
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(cfg['embed_path'])  
    
    #Step 4 - Encode Vector
    search_vector = encoder.encode(search_text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)

    #Step 5 - Search
    k = index.ntotal
    score, id = index.search(_vector, k=k)

    #Step 6 - Aggregate Results
    results = pd.DataFrame({'score': score[0], 'id': id[0]})
    merge = pd.merge(results, data, left_on='id', right_index=True)

    # Step 7 - Display Results
    print('-'*80, search_text, '-'*80, merge, '-'*80, sep='\n')

    v_text = ''
    
    for row in merge.head(result_number).itertuples():
        print(f'Body: {row.Body}')
        print(f'Score: {row.score} \n')
        v_text = v_text + str(row.Body) + ' '
    
    return v_text

def llm_summary():
   
    llm = Llama(model_path=cfg['model_path'], n_ctx=4096, n_gpu_layers = 35, verbose= False) 

    query = sum_prompt + str(v_data)
    single_turn_prompt = f"GPT4 Correct User: {query}<|end_of_turn|>GPT4 Correct Assistant:"
    
    output = llm(single_turn_prompt, max_tokens=4096)
    text_content = output['choices'][0]['text']

    print('-'*80, text_content, '-'*80, output, '-'*80, sep ='\n')
    return str(output)


#-----------------------------------------------------------
#   Main
#-----------------------------------------------------------

create_embeddings()
search_text = 'What is NVIDIA\'s CUDA strategy?'
sum_prompt = 'You are an equity research analyst. Determine the answer to the following question. ' + search_text + '  Only include facts that are directly mentioned in the article. Make sure all details from the article are captured in your summary. Article: '
result_number = 5
v_data = vector_query()
llm_summary()
