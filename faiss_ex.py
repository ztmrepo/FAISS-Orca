import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print('-'*80)

#Step 1 - Define Data
data = [['Where are your headquarters located?', 'location'],
['Throw my cellphone in the water', 'random'],
['Network Access Control?', 'networking'],
['Address', 'location']]

df = pd.DataFrame(data, columns = ['text', 'category'])

#Step 2 - Create Vectors from Text
text = df['text']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(text)

#Step 3 - Build FAISS Index
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)
faiss.normalize_L2(vectors)
index.add(vectors)

#Step 4 - Create a Vector Search
search_text = 'where is your office?'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])
faiss.normalize_L2(_vector)

#Step 5 - Search
k = index.ntotal
distances, ann = index.search(_vector, k=k)

#Step 6 - Display Results
results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
merge = pd.merge(results, df, left_on='ann', right_index=True)
print(merge)
print('-'*80)
