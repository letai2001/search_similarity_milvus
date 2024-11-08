from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from milvus_crud import MilvusCollection
from es_client import ElasticsearchClient
from vectorizer import Vectorizer
import asyncio
import json
from pyvi import ViTokenizer

app = FastAPI()

vectorizer = Vectorizer()  
milvus_collection_keyword = MilvusCollection(collection_name="text_vectors", dim=100)
milvus_collection_para = MilvusCollection(collection_name="text_st" , dim=768)
es_client = ElasticsearchClient()

class TextRequest(BaseModel):
    text: str

async def search_by_keywords(milvus_collection, es_client, text):
    """
    Tìm kiếm các bài viết có chứa từ khóa tương tự trong Milvus và Elasticsearch.

    :param milvus_collection: Đối tượng MilvusCollection
    :param es_client: Đối tượng ElasticsearchClient
    :param text: Văn bản đầu vào cần tìm kiếm từ khóa tương tự
    :return: Danh sách các bài viết từ Elasticsearch phù hợp với các từ khóa tìm được
    """
    text = ViTokenizer.tokenize(text)
    keywords = text.split()  # Tách các từ đã được tokenized
    found_keywords = set()

    for keyword in keywords:
        keyword_vector = milvus_collection.query_by_id(keyword)
        
        if keyword_vector is not None:
            similar_results = milvus_collection.search_vectors([keyword_vector], top_k=5)
            for result in similar_results[0]["results"]:
                found_keywords.add(result["id"]) 

    es_results = es_client.query_word(list(found_keywords))
    
    return [{"id": res[0], "content": res[1]} for res in es_results]

@app.post("/search")
async def search_text(request: TextRequest):
    # Cách 2: Vector hóa văn bản và tìm kiếm tương đồng trong Milvus
    input_vector = vectorizer.vectorize(request.text)
    similar_vectors = milvus_collection_para.search_vectors([input_vector], top_k=5)

    # Tạo kết quả ban đầu từ Cách 2
    initial_results = [{"id": result["id"], "distance": result["distance"]} for result in similar_vectors[0]["results"]]

    # Hàm generator để trả về kết quả theo stream
    async def result_stream():
        # Bước 1: Trả về kết quả từ Cách 2 ngay lập tức
        yield json.dumps({"method": "vector_search", "results": initial_results}) + "\n"
        
        # Bước 2: Chờ đợi và tìm kiếm từ khóa theo Cách 1
        keyword_results = await search_by_keywords(request.text)
        yield json.dumps({"method": "keyword_search", "results": keyword_results}) + "\n"

    # Trả về StreamingResponse để trả từng phần kết quả
    return StreamingResponse(result_stream(), media_type="application/json")
