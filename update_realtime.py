from kafka import KafkaConsumer
from json import loads
from es_client import ElasticsearchClient
from vectorizer import Vectorizer
from milvus_crud import MilvusCollection

KAFKA_TOPIC = "post_updates"
KAFKA_SERVERS = ["localhost:9092"]

def consume_and_process_updates():
    """
    Lắng nghe các sự kiện từ Kafka khi có bài viết mới hoặc cập nhật.
    Vector hóa nội dung bài viết và chèn vào Milvus.
    """
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='milvus_update_group',
        value_deserializer=lambda x: loads(x.decode('utf-8'))
    )

    es_client = ElasticsearchClient()
    vectorizer = Vectorizer()
    milvus_collection = MilvusCollection(collection_name="text_st", dim=768)

    print("Đang lắng nghe sự kiện từ Kafka...")

    for message in consumer:
        event_data = message.value
        post_id = event_data.get("post_id")

        if not post_id:
            print("Sự kiện không hợp lệ: thiếu post_id")
            continue

        print(f"Nhận sự kiện update cho bài viết với ID: {post_id}")

        post_data = es_client.query_by_id([post_id])

        if post_data:
            content = post_data[0].get("content", "")
            if content:
                vector = vectorizer.vectorize(content)
                
                milvus_collection.insert_data([post_id], [vector])
                print(f"Đã vector hóa và chèn bài viết với ID: {post_id} vào Milvus.")
            else:
                print(f"Nội dung của bài viết với ID: {post_id} trống.")
        else:
            print(f"Bài viết với ID: {post_id} không tồn tại trong Elasticsearch.")

if __name__ == "__main__":
    consume_and_process_updates()
