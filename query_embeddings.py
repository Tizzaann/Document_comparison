import os
import sys
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('.')
from create_embeddings import RetroMAEEmbedder, clean_filename, extract_title_from_url


class SemanticSearcher:
    """
    Класс для семантического поиска по документам Wikipedia.
    
    Выполняет поиск наиболее релевантных фрагментов текста на основе
    векторного сходства между запросом и содержимым документов.
    """
    
    def __init__(self, embeddings_dir="wikipedia_embeddings"):
        """
        Инициализация семантического поисковика.
        """
        self.embeddings_dir = embeddings_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.query_embedder = RetroMAEEmbedder(
            model_name="Shitao/RetroMAE",
            device=self.device,
            embedding_strategy="concat"
        )
    
    def url_to_filename(self, url):
        """
        Преобразование URL Wikipedia в соответствующее имя файла эмбеддингов.
        """
        title = extract_title_from_url(url)
        safe_title = clean_filename(title)
        
        for filename in os.listdir(self.embeddings_dir):
            if filename.startswith(f"{safe_title}_") and filename.endswith("_embeddings.npz"):
                return os.path.join(self.embeddings_dir, filename)
        
        return None
    
    def load_embeddings(self, url):
        """
        Загрузка эмбеддингов для заданного URL Wikipedia.
        """
        filename = self.url_to_filename(url)
        
        if not filename or not os.path.exists(filename):
            return {"error": f"No embeddings found for URL: {url}"}
        
        try:
            data = np.load(filename, allow_pickle=True)
            
            result = {
                'title': str(data['title']),
                'url': str(data['url']),
                'embeddings': data['embeddings'],
                'chunks_metadata': data['chunks_metadata'],
                'embedding_strategy': str(data['embedding_strategy'])
            }
            
            return result
        except Exception as e:
            return {"error": f"Error loading embeddings for {url}: {str(e)}"}
    
    def get_query_embedding(self, query_text):
        """
        Генерация эмбеддинга для текста запроса.
        """
        embedding = self.query_embedder.get_embeddings([query_text])
        return embedding[0]
    
    def find_relevant_chunks(self, query_embedding, doc_embeddings, top_k=3):
        """
        Поиск наиболее релевантных фрагментов для запроса.
        """
        if 'error' in doc_embeddings:
            return [{"error": doc_embeddings['error']}]
        
        similarities = cosine_similarity([query_embedding], doc_embeddings['embeddings'])[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk_metadata = doc_embeddings['chunks_metadata'][idx]
            
            item = {
                'text': chunk_metadata['text'],
                'metadata': chunk_metadata['metadata'],
                'similarity_score': float(similarities[idx])
            }
            
            results.append(item)
        
        return results
    
    def search(self, urls, query_text, top_k=3):
        """
        Поиск наиболее релевантных фрагментов по указанным URL.
        """
        query_embedding = self.get_query_embedding(query_text)
        
        results = {}
        for url in urls:
            doc_embeddings = self.load_embeddings(url)
            relevant_chunks = self.find_relevant_chunks(query_embedding, doc_embeddings, top_k)
            
            results[url] = {
                'title': doc_embeddings.get('title', extract_title_from_url(url)),
                'chunks': relevant_chunks
            }
        
        return results


def display_results(results):
    """
    Отображение результатов поиска в читаемом формате.
    """
    for url, data in results.items():
        if 'chunks' not in data or not data['chunks']:
            continue
            
        for i, chunk in enumerate(data['chunks']):
            if 'error' in chunk:
                continue
                
            if 'metadata' in chunk:
                metadata = chunk['metadata']
                section_level = metadata.get('section_level', 'unknown')
                
                if section_level != 'introduction':
                    section_heading = None
                    if section_level == 'h1' and 'h1' in metadata:
                        section_heading = metadata['h1']
                    elif section_level == 'h2' and 'h2' in metadata:
                        section_heading = metadata['h2']
                    elif section_level == 'h3' and 'h3' in metadata:
                        section_heading = metadata['h3']
                    elif section_level == 'h4' and 'h4' in metadata:
                        section_heading = metadata['h4']
                    elif section_level == 'h5' and 'h5' in metadata:
                        section_heading = metadata['h5']
                    elif section_level == 'h6' and 'h6' in metadata:
                        section_heading = metadata['h6']
            
            text = chunk['text']
            text_preview = text[:200] + ("..." if len(text) > 200 else "")