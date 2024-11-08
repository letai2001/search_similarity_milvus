def create_es_query(keyword):
    return {
        "query": {
            "match": {
                "content": keyword
            }
        }
    }
from pymilvus import connections, Collection, utility
from milvus_crud import MilvusCollection
words_collection  = MilvusCollection('text_st', dim  = 768)
ids = []
vectors = []

with open("contents_vectors.txt", "r") as file:
    for line in file:
        # Tách ID và vector từ mỗi dòng
        parts = line.strip().split()
        id = parts[0]
        vector = list(map(float, parts[1:]))  # Chuyển đổi các giá trị vector thành số thực

        # Thêm ID và vector vào danh sách
        ids.append(id)
        vectors.append(vector)
words_collection.insert_data(ids, vectors, batch_size=10000)