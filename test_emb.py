import os
from langchain.embeddings import HuggingFaceEmbeddings

def test_huggingface_embedding():
    # 第一次使用时，会自动下载模型
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 测试嵌入文本
    test_text = "这是一个测试句子，用于验证嵌入效果"
    
    # 生成嵌入向量
    embedding_vector = embedding.embed_query(test_text)
    
    print("嵌入向量长度:", len(embedding_vector))
    print("前5个嵌入值:", embedding_vector[:5])
    
    # 获取模型本地缓存路径
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    
    print("\n模型本地缓存目录:")
    print(cache_dir)
    
    # 列出已下载的模型
    print("\n已下载的模型:")
    for root, dirs, files in os.walk(cache_dir):
        for dir in dirs:
            if "all-MiniLM-L6-v2" in dir:
                print(os.path.join(root, dir))

if __name__ == "__main__":
    test_huggingface_embedding()