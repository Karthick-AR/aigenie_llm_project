#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_NAME

def get_embedding_model():
    
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME)