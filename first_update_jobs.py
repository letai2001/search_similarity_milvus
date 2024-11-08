from datetime import datetime
from es_client import ElasticsearchClient
from vectorizer import Vectorizer
from milvus_crud import MilvusCollection  # Import lớp MilvusCollection từ file milvus_collection.py

def check_and_update_data(milvus_collection, es_client, vectorizer, query_data_file="data_post.json"):
    """
    Luồng cập nhật dữ liệu cho collection Milvus:
    
    1. Kiểm tra số lượng bản ghi trong collection.
    2. Nếu số lượng bản ghi dưới 100:
       - Truy vấn các bài viết từ Elasticsearch trong khoảng thời gian nhất định.
       - Vector hóa nội dung bài viết bằng Vectorizer.
       - Chèn dữ liệu vào Milvus.
    """
    # Kiểm tra số lượng dữ liệu trong collection
    num_entities = milvus_collection.get_num_entities()
    print(f"Số lượng bản ghi hiện tại trong collection: {num_entities}")

    if num_entities < 100:
        print("Collection có ít hơn 100 bản ghi. Đang truy vấn dữ liệu từ Elasticsearch...")

        # Thiết lập khoảng thời gian truy vấn
        start_date = datetime.strptime("10-20-2024", "%m-%d-%Y")
        end_date = datetime.strptime("11-05-2024", "%m-%d-%Y")
        start_date_str = start_date.strftime("%m/%d/%Y 08:00:01")
        end_date_str = end_date.strftime("%m/%d/%Y 13:59:59")

        # Truy vấn các bài viết từ Elasticsearch
        records = es_client.query_keyword(start_date_str, end_date_str, query_data_file)

        # Chuẩn bị dữ liệu để chèn vào Milvus
        ids = []
        vectors = []
        for record in records:
            post_id = record["_id"]
            content = record["_source"].get("content", "")

            # Vector hóa content bằng Vectorizer
            vector = vectorizer.vectorize(content)
            ids.append(post_id)
            vectors.append(vector)

        # Chèn dữ liệu vào Milvus
        milvus_collection.insert_data(ids, vectors)
    else:
        print("Collection đã có đủ dữ liệu.")

# Hàm chính
if __name__ == "__main__":
    # Khởi tạo collection Milvus
    milvus_collection = MilvusCollection(collection_name="text_st", dim=768)

    # Khởi tạo Elasticsearch client và vectorizer
    es_client = ElasticsearchClient()
    vectorizer = Vectorizer()

    # Kiểm tra và cập nhật dữ liệu trong collection nếu cần
    check_and_update_data(milvus_collection, es_client, vectorizer)
