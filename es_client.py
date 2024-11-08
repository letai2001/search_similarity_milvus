from elasticsearch import Elasticsearch
import config
import json
from datetime import datetime, timedelta
class ElasticsearchClient:
    def __init__(self):
        self.client = Elasticsearch(config.ELASTICSEARCH_HOST)

    def fetch_data(self, query, size=1000):
        response = self.client.search(
            index=config.ELASTICSEARCH_INDEX,
            body=query,
            size=size
        )
        return [(hit["_id"], hit["_source"]["content"]) for hit in response["hits"]["hits"]]
    def query_keyword(self, start_date_str, end_date_str, query_data_file):
        try:
            body = {
                "query": {
                    "bool": {
                        "must": [
                            {"match_phrase": {"type": "media"}},
                            {"range": {"created_time": {"gte": start_date_str, "lte": end_date_str}}}
                        ]
                    }
                },
                "sort": [{"created_time": {"order": "asc"}}],
                "_source": ["content", "title", "created_time", "topic_id"]
            }

            result = self.client.search(index='posts', scroll='2m', size=100, body=body, request_timeout=2000)
            scroll_id = result['_scroll_id']
            scroll_size = len(result['hits']['hits'])

            records = []

            while scroll_size > 0:
                for hit in result['hits']['hits']:
                    if 'title' not in hit['_source']:
                        hit['_source']['title'] = []
                    if 'content' not in hit['_source']:
                        hit['_source']['content'] = []

                    records.append(hit)

                result = self.client.scroll(scroll_id=scroll_id, scroll='2m', request_timeout=2000)
                scroll_id = result['_scroll_id']
                scroll_size = len(result['hits']['hits'])

            with open(query_data_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=4)

            return records

        except Exception as e:
            print(f"An error occurred: {e}")
            with open(query_data_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)

            return []
    def query_by_id(self, query_ids):
        response = self.client.search(
        index="posts", 
        body={
            "query": {
                "ids": {
                    "values": query_ids
                }
            }
        }
    )

    # Xử lý kết quả từ Elasticsearch
        results = []
        for hit in response['hits']['hits']:
            post_id = hit['_id']
            content = hit['_source'].get('content', "No content available")
            results.append({
                "post_id": post_id,
                "content": content,
            })
        return results
    def query_word(self, keywords):
        """
        Truy vấn các bài viết có chứa ít nhất một từ khóa trong trường 'keyword'.

        :param keywords: Danh sách các từ khóa (list of strings)
        :return: Danh sách các tuple (id, content) cho các bài viết phù hợp
        """
        # Tạo truy vấn Elasticsearch với điều kiện 'should' để tìm các từ khóa
        body = {
            "query": {
                "bool": {
                    "should": [{"match": {"keyword": keyword}} for keyword in keywords],
                    "minimum_should_match": 1
                }
            },
            "_source": ["content"]
        }

        # Thực hiện truy vấn
        response = self.client.search(index=config.ELASTICSEARCH_INDEX, body=body, size=1000)

        # Lấy id và content của các bài viết phù hợp
        results = [(hit["_id"], hit["_source"].get("content", "No content available")) for hit in response["hits"]["hits"]]
        
        return results

if __name__ == '__main__':
    es_client = ElasticsearchClient()

    start_date = "10-20-2024"
    end_date = "11-05-2024"
    start_date = datetime.strptime(start_date, "%m-%d-%Y")
    end_date = datetime.strptime(end_date, "%m-%d-%Y")
    start_date_str = start_date.strftime("%m/%d/%Y 08:00:01")
    end_date_str = end_date.strftime("%m/%d/%Y 13:59:59")
    query_data_file = "data_post.json"
    print("Bat dau query!")
    data = es_client.query_keyword(start_date_str, end_date_str, query_data_file)
    print(len(data))
    print("Da query xong!")
