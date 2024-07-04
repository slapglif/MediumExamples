import spacy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from loguru import logger

logger.add(
    "semantic_search.log", format="{time} {level} {message}", level="INFO"
)

# 1. Load Dataset from Hugging Face Hub 

dataset = load_dataset("squad", split="train[:100]")  # Load a small subset of SQuAD

# Preprocess dataset to get a list of contexts 
knowledge_base = [sample['context'] for sample in dataset] 

# 2. Semantic Text Splitting with spaCy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

entity_chunks = []
for text in knowledge_base:
    entity_chunks.extend(extract_entities(text))

logger.info(f"🚀 Extracted {len(entity_chunks)} Entities") 

# 3. Matryoshka Embeddings with Sentence Transformers

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

paragraph_embeddings = model.encode(knowledge_base)
sentence_embeddings = model.encode(
    [sent.text for doc in nlp.pipe(knowledge_base) for sent in doc.sents]
)

logger.info("✅ Embeddings Generated")

# 4. Semantic Grouping with k-means

num_clusters = 5  # Adjust based on your data and desired granularity
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(sentence_embeddings)
cluster_labels = kmeans.labels_

logger.info(f"📊 Cluster Labels: {cluster_labels}")

# --- Example Retrieval & Distance Calculation ---

query = "What is the history of Google?"
query_embedding = model.encode([query])[0]

distances = euclidean_distances([query_embedding], sentence_embeddings)
closest_sentence_index = distances.argmin()

logger.info(f"🔍 Closest Sentence: {dataset[closest_sentence_index]['context']}")

# --- Log Distances (for illustration) ---
for i, embedding in enumerate(sentence_embeddings):
    distance = euclidean_distances([query_embedding], [embedding])
    logger.info(f"  Distance to Sentence {i+1}: {distance[0][0]:.2f}")
