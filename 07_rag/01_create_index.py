from utils.config import Config
from utils.llms import get_llm
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()

llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

# 使用嵌入模型实例化 内存向量 数据库实例，用于存储文档向量
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=llm_embedding,
    persist_directory="./chroma_langchain_db"
    #document_loader=PyPDFLoader,
)

# 1。加载文档
# 指定 PDF 文件的路径，这里文件名为”健康档案.pdf“,位于当前目录下
file_path = "./健康档案.pdf"

# 使用 PyPDFLoader 创建一个加载器，用于读取该 PDF 文件
loader = PyPDFLoader(file_path)

docs = loader.load()

# 取出列表中的第一个document, 通常对应 PDF的第一页
first_doc = docs[0]

# 2. 切分文档
# 创建递归字符文本分割器实例，设置切分参数
# 每个文档块的大小为 500 个字符
# 文档块之间重叠 100 个字符，确保上下文连贯性
# 添加起始索引，跟踪每个块在原文档中的位置
# 定义分隔符优先级列表，按从高到低的优先级依次尝试分割文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True,
    separators=[
        "\n\n",
          "\n",
        # 中文句号
        "。",
        # 中文感叹号
        "！",
        # 中文问号
        "？",
        # 英文感叹号
        "!",
        # 英文问号
        "?",
        # 英文句号
        ".",
        # 中文分号
        "；",
        # 英文分号
        ";",
        # 中文逗号
        "，",
        # 英文逗号
        ",",
        # 中文冒号
        "：",
        # 英文冒号
        ":",
        # 空格
        " ",
        # 优先级最低：空字符串（强制按字符切分）
        ""
    ]
)

# 使用文本分割器将文档切分成多个小块
all_splits = text_splitter.split_documents(docs)
# 打印切分后的文档块数量
print(f"######文档一共切分了 {len(all_splits)} 个文本块 \n")

# 3、创建索引并写入向量数据库
# 将切分后的文档块添加到向量数据库中，并返回每个文档的ID
document_ids = vector_store.add_documents(documents=all_splits)
# 打印前 3 个文档的 ID
print(f"#######前3个文档的ID：{document_ids[:3]}")


