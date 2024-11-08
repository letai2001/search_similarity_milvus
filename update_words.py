from pymilvus import connections, Collection, utility
from milvus_crud import MilvusCollection
import time
# Kết nối đến Milvus
# connections.connect("default", host="localhost", port="19530")

# collection_name = "text_st"

# # Kiểm tra xem collection đã tồn tại chưa
# if not utility.has_collection(collection_name):
#     print(f"Collection '{collection_name}' không tồn tại.")
# else:
#     collection = Collection(name=collection_name)
    
#     collection.load()
#     print(f"Collection '{collection_name}' đã được load thành công.")
def read_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)  
    return stopwords
    
    
def read_vectors(file_path, stopwords):
    vectors = []
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  
        for line in f:
            parts = line.split()
            word = parts[0]  
            if word not in stopwords:  
                try:
                    vector = list(map(float, parts[1:]))  
                    if len(vector) == 100:  
                        vectors.append((word, vector)) 
                    else:
                        print(f"Vector không đúng số chiều: {word}")
                except ValueError:
                    print(f"Không thể chuyển đổi vector: {word}")
    return vectors


if __name__ == '__main__':
    words_collection  = MilvusCollection('text_vectors', dim  = 100)
    file_path = "word2vec_vi_words_100dims.txt"
    file_path_stopwords = "vietnamese-stopwords-dash.txt"
    stopwords  = read_stopwords(file_path_stopwords)
    data_tuples  = read_vectors(file_path , stopwords)
    start_time = time.time()
    words_collection.insert_by_tuple(data_tuples, batch_size=100000)  # Chèn dữ liệu với batch_size là 500
    end_time = time.time()
    print(f"Thoi gian insert: {end_time - start_time}")