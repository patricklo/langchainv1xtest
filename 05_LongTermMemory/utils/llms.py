import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .logger import LoggerManager

logger = LoggerManager.get_logger()

MODEL_CONFIGS = {
    # 使用自定义 openai 代理服务的配置
    "openai": {
        #LLM服务的基础URLo
        "base_url":"https://apis.itedus.cn/v1/",
        "api_key":"sk-wKFKXI1pM5YPhzAt171eF6Ea43Cc4748Ba574497AbFaFf86",
        "chat_model":"gpt-5.5",
        #向量嵌入模型名称
        "embedding_model": "text-embedding-3-small"
    },
}

DEFAULT_LLM_TYPE = "openai"
DEFAULT_TEMPERATURE = 0

class LLMInitializationError(Exception):
    pass

def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型：{llm_type}")

        config = MODEL_CONFIGS[llm_type]

        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,
            max_retries=2
        )

        llm_embedding = OpenAIEmbeddings(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"],
        )

        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat, llm_embedding

    except ValueError as ve:
        # 记录详细的配置错误日志
        logger.error(f"LLM配置错误: {str(ve)}")
        # 将其包装成自定义异常并抛出，方便统一处理
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    # 捕获其他所有异常
    except Exception as e:
        # 记录通用的初始化失败日志
        logger.error(f"初始化LLM失败: {str(e)}")
        # 抛出自定义异常，供调用方判断与重试
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")

def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    try:
        return initialize_llm(llm_type)
    # 捕获自定义的初始化异常
    except LLMInitializationError as e:
        # 打印警告日志，说明会尝试使用默认配置重试
        logger.warning(f"使用默认配置重试: {str(e)}")
        # 如果当前类型不是默认类型，则退回到默认 LLM 类型再尝试一次
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        # 如果已经是默认类型仍然失败，则继续向上抛出异常
        raise
