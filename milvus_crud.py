from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

class MilvusCollection:
    def __init__(self, collection_name, dim, index_params=None):
        connections.connect("default", host="localhost", port="19530")

        self.collection_name = collection_name
        self.dim = dim

        self.index_params = index_params or {
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {
                "M": 8,  
                "efConstruction": 200  
            },
            "quantization_type": "PQ",
            "quantization_params": {
                "nbits": 8,  
            }
        }

        if utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name)
            self.collection.load()
            print(f"Collection '{self.collection_name}' đã tồn tại và được load vào bộ nhớ.")
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields, description="Collection lưu trữ các vector 768 chiều")
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.collection.create_index(field_name="vector", index_params=self.index_params)
            print(f"Collection '{self.collection_name}' đã được tạo mới và thiết lập index.")
    
    def insert_data(self, ids, vectors, batch_size = 10000):
        """
        Thêm các vector vào collection theo batch để tăng tốc độ chèn.

        :param ids: Danh sách các id (string) cho mỗi vector
        :param vectors: Danh sách các vector (list of lists) có kích thước bằng `dim`
        :param batch_size: Kích thước của mỗi batch
        """
        if len(ids) != len(vectors):
            raise ValueError("Số lượng ids và vectors phải bằng nhau.")
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            self.collection.insert([batch_ids, batch_vectors])
            print(f"Đã chèn batch {i // batch_size + 1} với {len(batch_ids)} bản ghi vào collection '{self.collection_name}'.")

        self.collection.flush()  
        print(f"Đã chèn tổng cộng {len(ids)} vector vào collection '{self.collection_name}'.")

    def insert_by_tuple(self, data_tuples, batch_size=1000):
        """
        Thêm dữ liệu vào collection từ danh sách các tuple (id, vector) theo batch.

        :param data_tuples: Danh sách các tuple, mỗi tuple gồm (id, vector)
        :param batch_size: Kích thước của mỗi batch
        """
        ids = [item[0] for item in data_tuples]
        vectors = [item[1] for item in data_tuples]

        self.insert_data(ids, vectors, batch_size=batch_size)
    def query_by_id(self, id_string):
        """
        Truy xuất vector của một id cụ thể trong Milvus collection.

        :param id_string: Chuỗi id cần tìm (có thể là một từ)
        :return: Vector tương ứng của id, hoặc None nếu không tìm thấy
        """
        # Sử dụng biểu thức `expr` để truy vấn theo `id`
        expr = f"id == '{id_string}'"

        # Thực hiện truy vấn trên Milvus collection
        results = self.collection.query(expr=expr, output_fields=["vector"])
        
        if results:
            # Trả về vector của id nếu tìm thấy
            return results[0]["vector"]
        else:
            print(f"Không tìm thấy vector cho id: {id_string}")
            return None

    def search_vectors(self, query_vectors, top_k=5):
        """
        Tìm kiếm các vector gần nhất trong collection và trả về danh sách id tương ứng.

        :param query_vectors: Danh sách các vector query để tìm kiếm
        :param top_k: Số lượng kết quả gần nhất cần lấy
        :return: Danh sách các id tương ứng
        """
        search_params = {"metric_type": "L2", "params": {"ef": 50}}

        # Thực hiện tìm kiếm
        results = self.collection.search(
            data=query_vectors,
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["id"]
        )

        # Chỉ lấy danh sách các id
        search_output = []
        for result in results:
            ids = [hit.entity.get("id") for hit in result]
            search_output.extend(ids)  # Thêm các id vào danh sách kết quả
        
        return search_output
    def get_num_entities(self):
        return self.collection.num_entities

    def delete_collection(self):
        """Xóa collection khỏi Milvus."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' đã được xóa.")
        else:
            print(f"Collection '{self.collection_name}' không tồn tại.")





if __name__ == "__main__":
    milvus_collection = MilvusCollection(collection_name="text_st", dim=768)

    ids = ["vec1", "vec2", "vec3"]
    vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    milvus_collection.insert_data(ids, vectors)

    query_vector = [[0.15] * 768]
    search_results = milvus_collection.search_vectors(query_vectors=query_vector, top_k=2)
    print("Kết quả tìm kiếm:", search_results)
