from transformers import AutoModel, AutoTokenizer
import torch
import time 
from pyvi import ViTokenizer
from FlagEmbedding import BGEM3FlagModel

class Vectorizer:
    def __init__(self, model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    # def __init__(self, model_name="BAAI/bge-m3"):
    #     # Khởi tạo model BGEM3FlagModel với tùy chọn `use_fp16=True`
    #     self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def vectorize(self, text):
        text = ViTokenizer.tokenize(text)
        # print(text)
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.squeeze().cpu().numpy()
    # def vectorize(self, text):
    #     # Sử dụng `encode` để lấy embedding từ `dense_vecs`
    #     embeddings = torch.tensor(self.model.encode([text])['dense_vecs'])
    #     return embeddings[0]  # Trả về embedding của văn bản đầu tiên

if __name__ == "__main__":
    vecto = Vectorizer()
    text =  """Người dân địa phương đã cứu hộ thành công một bé gái 5 tuổi bị lạc trong rừng suốt hai ngày. Lực lượng cứu hộ đã phối hợp với các cơ quan chức năng và người dân quanh vùng để tìm kiếm và đưa bé về nhà an toàn."""
    start_time = time.time()  
    vector_output =  vecto.vectorize(text)
    end_time = time.time()
    print(vector_output)
    print(f"Thời gian thêm tất cả vector vào FAISS: {end_time - start_time} giây")
