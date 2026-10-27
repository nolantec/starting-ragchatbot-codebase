import os
from dataclasses import dataclass
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

@dataclass
class Config:
    """RAG 系统配置"""
    # DeepSeek API 设置（OpenAI 兼容格式）
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # 嵌入模型设置
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # 文档处理设置
    CHUNK_SIZE: int = 800       # 向量存储文本块大小
    CHUNK_OVERLAP: int = 100     # 块之间重叠字符数
    MAX_RESULTS: int = 5         # 最大搜索结果数
    MAX_HISTORY: int = 2         # 对话历史保留轮数

    # 数据库路径
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB 存储位置

config = Config()
