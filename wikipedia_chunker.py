import requests
from bs4 import BeautifulSoup
import re


class WikipediaChunker:
    """
    Класс для разбиения контента Wikipedia на смысловые фрагменты.
    
    Позволяет извлекать текст из статей Wikipedia и разбивать его на параграфы
    с сохранением иерархической структуры заголовков и метаданных.
    """
    
    def __init__(self, min_paragraph_chars=100, max_paragraph_chars=3000, combine_small_paragraphs=True):
        """
        Инициализация парсера Wikipedia с параметрами конфигурации.
        """
        self.min_paragraph_chars = min_paragraph_chars
        self.max_paragraph_chars = max_paragraph_chars
        self.combine_small_paragraphs = combine_small_paragraphs
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_chunks_from_url(self, url):
        """
        Обработка одного URL Wikipedia и возврат фрагментов на уровне параграфов.
        """
        try:
            soup = self._get_wikipedia_content(url)
            result = self._extract_paragraph_chunks(soup)
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def get_chunks_from_multiple_urls(self, urls):
        """
        Обработка нескольких URL Wikipedia и возврат их фрагментов.
        """
        results = {}
        for url in urls:
            results[url] = self.get_chunks_from_url(url)
        return results
    
    def _get_wikipedia_content(self, url):
        """
        Получение контента со страницы Wikipedia и возврат объекта BeautifulSoup.
        """
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    
    def _split_large_paragraph(self, paragraph_text):
        """
        Разбиение большого параграфа на меньшие фрагменты без нарушения предложений.
        """
        if len(paragraph_text) <= self.max_paragraph_chars:
            return [paragraph_text]
        
        sentences = re.split(r'(?<=[.!?])\s+', paragraph_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_paragraph_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_paragraph_chunks(self, soup):
        """
        Извлечение контента со страницы Wikipedia и организация в фрагменты уровня параграфов
        с полными иерархическими метаданными.
        """
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
        
        title = soup.find('h1', {'id': 'firstHeading'}).text.strip()
        
        main_div = content_div.find('div', {'class': 'mw-parser-output'})
        if not main_div:
            main_div = content_div
        
        result = {
            'title': title,
            'chunks': []
        }
        
        current_h1 = None
        current_h2 = None
        current_h3 = None
        current_h4 = None
        current_h5 = None
        current_h6 = None
        in_introduction = True
        
        current_section_paragraphs = []
        current_section_metadata = None
        
        elements = main_div.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        
        for element in elements:
            tag_name = element.name
            
            if element.find_parent('div', {'id': 'toc'}) or \
               element.find_parent('div', {'class': 'navbox'}) or \
               element.find_parent('table') or \
               element.find_parent('div', {'class': 'reflist'}):
                continue
            
            if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if current_section_paragraphs:
                    self._process_section_paragraphs(current_section_paragraphs, current_section_metadata, result)
                    current_section_paragraphs = []
                
                heading_text = element.get_text().strip()
                
                if tag_name == 'h1':
                    current_h1 = heading_text
                    current_h2 = current_h3 = current_h4 = current_h5 = current_h6 = None
                    in_introduction = False
                elif tag_name == 'h2':
                    current_h2 = heading_text
                    current_h3 = current_h4 = current_h5 = current_h6 = None
                    in_introduction = False
                elif tag_name == 'h3':
                    current_h3 = heading_text
                    current_h4 = current_h5 = current_h6 = None
                    in_introduction = False
                elif tag_name == 'h4':
                    current_h4 = heading_text
                    current_h5 = current_h6 = None
                    in_introduction = False
                elif tag_name == 'h5':
                    current_h5 = heading_text
                    current_h6 = None
                    in_introduction = False
                elif tag_name == 'h6':
                    current_h6 = heading_text
                    in_introduction = False
            
            elif tag_name == 'p':
                text = element.get_text().strip()
                if not text:
                    continue
                
                metadata = {
                    'title': title,
                    'section_level': 'introduction' if in_introduction else None,
                    'h1': current_h1,
                    'h2': current_h2,
                    'h3': current_h3,
                    'h4': current_h4,
                    'h5': current_h5,
                    'h6': current_h6,
                    'element_type': tag_name
                }
                
                if current_h6:
                    metadata['section_level'] = 'h6'
                elif current_h5:
                    metadata['section_level'] = 'h5'
                elif current_h4:
                    metadata['section_level'] = 'h4'
                elif current_h3:
                    metadata['section_level'] = 'h3'
                elif current_h2:
                    metadata['section_level'] = 'h2'
                elif current_h1:
                    metadata['section_level'] = 'h1'
                
                if self.combine_small_paragraphs:
                    if not current_section_metadata:
                        current_section_metadata = metadata.copy()
                    current_section_paragraphs.append(text)
                else:
                    if len(text) >= self.min_paragraph_chars:
                        paragraph_chunks = self._split_large_paragraph(text)
                        for i, paragraph_text in enumerate(paragraph_chunks):
                            chunk = {
                                'text': paragraph_text,
                                'metadata': metadata.copy()
                            }
                            if len(paragraph_chunks) > 1:
                                chunk['metadata']['part'] = f"{i+1}/{len(paragraph_chunks)}"
                            result['chunks'].append(chunk)
        
        if current_section_paragraphs:
            self._process_section_paragraphs(current_section_paragraphs, current_section_metadata, result)
        
        result['chunks'] = [chunk for chunk in result['chunks'] 
                           if len(chunk['text']) >= self.min_paragraph_chars]
        
        return result
    
    def _process_section_paragraphs(self, paragraphs, metadata, result):
        """
        Обработка параграфов из секции с объединением малых фрагментов при необходимости.
        """
        if not paragraphs:
            return
        
        valid_paragraphs = []
        combined_text = ""
        
        for text in paragraphs:
            if len(text) < self.min_paragraph_chars and self.combine_small_paragraphs:
                combined_text += " " + text if combined_text else text
            else:
                valid_paragraphs.append(text)
        
        if combined_text and len(combined_text) >= self.min_paragraph_chars:
            valid_paragraphs.append(combined_text)
        
        for text in valid_paragraphs:
            paragraph_chunks = self._split_large_paragraph(text)
            for i, paragraph_text in enumerate(paragraph_chunks):
                if len(paragraph_text) >= self.min_paragraph_chars:
                    chunk = {
                        'text': paragraph_text,
                        'metadata': metadata.copy()
                    }
                    if len(paragraph_chunks) > 1:
                        chunk['metadata']['part'] = f"{i+1}/{len(paragraph_chunks)}"
                    result['chunks'].append(chunk)