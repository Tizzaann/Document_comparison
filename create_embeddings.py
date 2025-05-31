import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from wikipedia_chunker import WikipediaChunker
import numpy as np
import os
import urllib.parse
import re


class RetroMAEEmbedder:
    """
    Класс для создания эмбеддингов текста с использованием модели RetroMAE.
    
    Поддерживает различные стратегии комбинирования CLS и OT эмбеддингов
    для получения качественных векторных представлений текста.
    """
    
    def __init__(self, model_name="Shitao/RetroMAE", device=None, embedding_strategy="concat"):
        """
        Инициализация эмбеддера RetroMAE.
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_strategy = embedding_strategy
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        if embedding_strategy == "weighted":
            self.alpha = 0.5
            self.cls_dim = self.model.config.hidden_size
            self.ot_dim = self.cls_dim
            self.output_dim = self.cls_dim
            
            self.projection_layer_cls = nn.Linear(self.cls_dim, self.output_dim).to(self.device)
            self.projection_layer_ot = nn.Linear(self.ot_dim, self.output_dim).to(self.device)
    
    def get_optimal_transport_embedding(self, token_embeddings, attention_mask):
        """
        Вычисление эмбеддинга методом оптимального транспорта из токенных эмбеддингов.
        """
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def get_embeddings(self, texts, batch_size=8):
        """
        Получение эмбеддингов для списка текстов.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            
            cls_embedding = outputs.last_hidden_state[:, 0]
            
            ot_embedding = self.get_optimal_transport_embedding(
                outputs.last_hidden_state,
                encoded_input["attention_mask"]
            )
            
            if self.embedding_strategy == "concat":
                combined_embedding = torch.cat([cls_embedding, ot_embedding], dim=1)
            elif self.embedding_strategy == "weighted":
                projected_cls = self.projection_layer_cls(cls_embedding)
                projected_ot = self.projection_layer_ot(ot_embedding)
                combined_embedding = self.alpha * projected_cls + (1 - self.alpha) * projected_ot
            else:
                raise ValueError(f"Unknown embedding strategy: {self.embedding_strategy}")
            
            embeddings_np = combined_embedding.cpu().numpy()
            all_embeddings.append(embeddings_np)
        
        return np.vstack(all_embeddings)

    def process_wikipedia_chunks(self, result_data):
        """
        Обработка фрагментов из Wikipedia и генерация эмбеддингов.
        """
        if 'error' in result_data:
            return {'error': result_data['error']}
        
        texts = [chunk['text'] for chunk in result_data['chunks']]
        embeddings = self.get_embeddings(texts)
        
        result = {
            'title': result_data['title'],
            'embeddings_data': []
        }
        
        for i, (chunk, embedding) in enumerate(zip(result_data['chunks'], embeddings)):
            result['embeddings_data'].append({
                'chunk': chunk,
                'embedding': embedding
            })
        
        return result


def clean_filename(filename):
    """
    Создание безопасного имени файла из заголовка Wikipedia или URL.
    """
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", filename)
    safe_name = safe_name.replace(" ", "_")
    safe_name = re.sub(r'_+', "_", safe_name)
    return safe_name


def extract_title_from_url(url):
    """
    Извлечение чистого заголовка из URL Wikipedia.
    """
    parsed_url = urllib.parse.urlparse(url)
    path = parsed_url.path
    
    if path.endswith('/'):
        path = path[:-1]
    
    title = os.path.basename(path)
    title = urllib.parse.unquote(title)
    title = title.replace('_', ' ')
    
    return title