import os
import sys
import numpy as np
import requests
import argparse

sys.path.append('.')
from create_embeddings import RetroMAEEmbedder, clean_filename, extract_title_from_url
from query_embeddings import SemanticSearcher, display_results
from wikipedia_chunker import WikipediaChunker


class AutomatedDocumentComparer:
    """Автоматический компаратор документов с использованием локальной LLM через Ollama"""
    
    def __init__(self, model_name="gemma3:27b", temperature=0.0, embeddings_dir="wikipedia_embeddings"):
        """
        Инициализация компаратора документов
        
        Args:
            model_name (str): Название модели Ollama
            temperature (float): Параметр температуры для генерации текста
            embeddings_dir (str): Директория для хранения эмбеддингов
        """
        self.model_name = model_name
        self.temperature = temperature
        self.api_url = "http://localhost:11434/api/generate"
        self.embeddings_dir = embeddings_dir
        
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.chunker = WikipediaChunker(
            min_paragraph_chars=200,
            max_paragraph_chars=3000,
            combine_small_paragraphs=True
        )
        
        self.embedder = RetroMAEEmbedder(
            model_name="Shitao/RetroMAE",
            embedding_strategy="concat"
        )
        
        self.searcher = SemanticSearcher(embeddings_dir=self.embeddings_dir)
    
    def check_embeddings_exist(self, url):
        """
        Проверка существования эмбеддингов для указанного URL
        
        Args:
            url (str): URL Wikipedia для проверки
            
        Returns:
            str or None: Путь к файлу эмбеддингов или None
        """
        title = extract_title_from_url(url)
        safe_title = clean_filename(title)
        
        for filename in os.listdir(self.embeddings_dir):
            if filename.startswith(f"{safe_title}_") and filename.endswith("_embeddings.npz"):
                return os.path.join(self.embeddings_dir, filename)
        
        return None
    
    def create_embeddings_for_url(self, url):
        """
        Создание эмбеддингов для одного URL
        
        Args:
            url (str): URL Wikipedia для обработки
            
        Returns:
            dict: Результат создания эмбеддингов
        """
        chunks_result = self.chunker.get_chunks_from_url(url)
        
        if 'error' in chunks_result:
            return {'error': f"Error chunking {url}: {chunks_result['error']}"}
        
        embeddings_result = self.embedder.process_wikipedia_chunks(chunks_result)
        
        if 'error' in embeddings_result:
            return {'error': f"Error creating embeddings for {url}: {embeddings_result['error']}"}
        
        if 'embeddings_data' in embeddings_result:
            embeddings = np.array([item['embedding'] for item in embeddings_result['embeddings_data']])
            
            chunks_metadata = []
            for item in embeddings_result['embeddings_data']:
                chunks_metadata.append(item['chunk'])
            
            save_data = {
                'title': embeddings_result['title'],
                'url': url,
                'embeddings': embeddings,
                'chunks_metadata': chunks_metadata,
                'embedding_strategy': 'concat'
            }
            
            if 'title' in embeddings_result:
                filename = clean_filename(embeddings_result['title'])
            else:
                filename = clean_filename(extract_title_from_url(url))
            
            output_file = os.path.join(self.embeddings_dir, f"{filename}_concat_embeddings.npz")
            
            np.savez(output_file, **save_data)
            
            return {
                'success': True,
                'title': embeddings_result['title'],
                'chunks_count': len(embeddings_result['embeddings_data']),
                'file_path': output_file
            }
        
        return {'error': 'No embeddings data generated'}
    
    def ensure_embeddings_exist(self, urls):
        """
        Обеспечение существования эмбеддингов для всех указанных URL
        
        Args:
            urls (list): Список URL Wikipedia
            
        Returns:
            dict: Статус эмбеддингов для каждого URL
        """
        results = {}
        
        for url in urls:
            existing_file = self.check_embeddings_exist(url)
            
            if existing_file:
                results[url] = {
                    'status': 'exists',
                    'file_path': existing_file
                }
            else:
                creation_result = self.create_embeddings_for_url(url)
                
                if 'error' in creation_result:
                    results[url] = {
                        'status': 'error',
                        'error': creation_result['error']
                    }
                else:
                    results[url] = {
                        'status': 'created',
                        'title': creation_result['title'],
                        'chunks_count': creation_result['chunks_count'],
                        'file_path': creation_result['file_path']
                    }
        
        return results
    
    def query_ollama(self, prompt):
        """
        Запрос к локальной модели Ollama
        
        Args:
            prompt (str): Промпт для отправки модели
            
        Returns:
            str: Сгенерированный ответ
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature}
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                return f"Error: API returned status code {response.status_code}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def create_comparison_prompt(self, query, doc1_results, doc2_results):
        """
        Создание промпта для LLM для сравнения информации из двух документов
        
        Args:
            query (str): Исходный пользовательский запрос
            doc1_results (dict): Результаты из первого документа
            doc2_results (dict): Результаты из второго документа
            
        Returns:
            str: Отформатированный промпт для LLM
        """
        prompt = f"""You are a helpful assistant that creates precise comparisons between documents. 

USER QUERY: {query}

DOCUMENT 1: {doc1_results['title']}
{'-' * 40}
"""
        
        for i, chunk in enumerate(doc1_results['chunks']):
            if 'error' not in chunk:
                section_info = ""
                if 'metadata' in chunk:
                    metadata = chunk['metadata']
                    section_level = metadata.get('section_level', 'unknown')
                    
                    if section_level == 'introduction':
                        section_info = "Section: Introduction\n"
                    elif section_level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and section_level in metadata:
                        section_info = f"Section: {metadata[section_level]}\n"
                
                prompt += f"\nCHUNK {i+1} (Similarity: {chunk['similarity_score']:.4f}):\n{section_info}{chunk['text']}\n"
        
        prompt += f"\n{'-' * 40}\n\nDOCUMENT 2: {doc2_results['title']}\n{'-' * 40}\n"
        
        for i, chunk in enumerate(doc2_results['chunks']):
            if 'error' not in chunk:
                section_info = ""
                if 'metadata' in chunk:
                    metadata = chunk['metadata']
                    section_level = metadata.get('section_level', 'unknown')
                    
                    if section_level == 'introduction':
                        section_info = "Section: Introduction\n"
                    elif section_level in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'] and section_level in metadata:
                        section_info = f"Section: {metadata[section_level]}\n"
                
                prompt += f"\nCHUNK {i+1} (Similarity: {chunk['similarity_score']:.4f}):\n{section_info}{chunk['text']}\n"
        
        prompt += f"""
{'-' * 40}

Based on the information above, please provide:

1. A concise summary of how the two documents answer the query: "{query}"
2. The key differences in how each document addresses the query
3. Any unique information provided by each document that the other doesn't mention
4. A brief conclusion on which document provides more comprehensive or relevant information on this topic

Focus on substantive differences in content, not just writing style or formatting.
Give your answer in russian language.
"""
        
        return prompt
    
    def compare_documents(self, urls, query, top_k=3):
        """
        Автоматическое сравнение документов из двух URL с созданием эмбеддингов при необходимости
        
        Args:
            urls (list): Список из двух URL Wikipedia для сравнения
            query (str): Текст запроса
            top_k (int): Количество топ результатов на документ
            
        Returns:
            dict: Словарь с результатами поиска и сгенерированным сравнением
        """
        if len(urls) != 2:
            return {"error": "Exactly two URLs must be provided for comparison."}
        
        embeddings_status = self.ensure_embeddings_exist(urls)
        
        for url, status in embeddings_status.items():
            if status['status'] == 'error':
                return {"error": f"Failed to create embeddings for {url}: {status['error']}"}
        
        search_results = self.searcher.search(urls, query, top_k)
        
        doc1_results = search_results[urls[0]]
        doc2_results = search_results[urls[1]]
        
        comparison_prompt = self.create_comparison_prompt(query, doc1_results, doc2_results)
        comparison = self.query_ollama(comparison_prompt)
        
        return {
            "embeddings_status": embeddings_status,
            "search_results": search_results,
            "comparison": comparison,
            "prompt_used": comparison_prompt
        }


def main():
    """Основная функция для обработки аргументов командной строки и выполнения сравнения"""
    parser = argparse.ArgumentParser(description="Automatically compare information from two Wikipedia documents.")
    parser.add_argument("--urls", nargs=2, required=True, help="Two Wikipedia URLs to compare")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results per document")
    parser.add_argument("--model", default="gemma3:27b", help="Ollama model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for text generation")
    parser.add_argument("--embeddings_dir", default="wikipedia_embeddings", help="Directory for embeddings")
    parser.add_argument("--save_prompt", action="store_true", help="Save the generated prompt to a file")
    parser.add_argument("--verbose", action="store_true", help="Show detailed search results")
    
    args = parser.parse_args()
    
    comparer = AutomatedDocumentComparer(
        model_name=args.model, 
        temperature=args.temperature,
        embeddings_dir=args.embeddings_dir
    )
    
    results = comparer.compare_documents(args.urls, args.query, args.top_k)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print("=== EMBEDDINGS STATUS ===")
    for url, status in results["embeddings_status"].items():
        title = extract_title_from_url(url)
        if status['status'] == 'exists':
            print(f"{title}: Embeddings already existed")
        elif status['status'] == 'created':
            print(f"{title}: Embeddings created ({status['chunks_count']} chunks)")
        else:
            print(f"{title}: Error - {status.get('error', 'Unknown error')}")
    
    if args.verbose:
        print("\n=== SEARCH RESULTS ===")
        display_results(results["search_results"])
    
    print("\n=== DOCUMENT COMPARISON ===")
    print(results["comparison"])
    
    if args.save_prompt:
        with open("comparison_prompt.txt", "w", encoding="utf-8") as f:
            f.write(results["prompt_used"])
        print(f"\nPrompt saved to comparison_prompt.txt")


if __name__ == "__main__":
    main()