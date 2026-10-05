import os
from langchain_openai import ChatOpenAI,OpenAIEmbeddings


from .logger import LoggerManager

#获取全局日志实例，用于在工具加载和调用过程中记录日志
logger = LoggerManager.get_logger()

#定义不同LLM 类型对应的模型与接口配置
MODEL_CONFIGS = {
    #使用自定义openai代理服务的配置
    "openai":{
        #LLM服务的基础URL
        "base_url":"https://apis.itedus.cn/v1/",
        "api_key":"sk-F9pzJTFWPeaB3FV1F821DbD7E29041F482860215498a30Bf",
        "chat_model":"gpt-4o",
        #向量嵌入模型名称
        "embedding_model": "text-embedding-3-small"
    }
}

DEFAULT_LLM_TYPE = "openai"
DEFAULT_TEMPERATURE = 0

#自定义异常类， 在LLM初始化失败时统一抛出该异常
class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass

#定义函数用于初始化LLM与Embedding实例，并返回二者
def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI,OpenAIEmbeddings]:
    """
    初始化LLM实例
    args:
        llm_type (str): LLM类型，可选值 为‘openai'

    Returns:
        ChatOpenAI: 初始化后的LLM实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持LLM类型：{llm_type}")

        config = MODEL_CONFIGS[llm_type]

        #创建对话LLM实例
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

        logger.info(f"初始化成功，{llm_type}")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误:{str(ve)}")
        raise LLMInitializationError(f"LLM配置错误:{str(ve)}")
    except Exception as e:
        # 记录通用的初始化失败日志
        logger.error(f"初始化LLM失败: {str(e)}")
        # 抛出自定义异常，供调用方判断与重试
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")

def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    args: llm_type (str): LLM类型

    Returns: ChatOpenAI LLM实例
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        # 打印警告日志，说明会尝试使用默认配置重试
        logger.warning(f"使用默认配置重试: {str(e)}")
        # 如果当前类型不是默认类型，则退回到默认 LLM 类型再尝试一次
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        # 如果已经是默认类型仍然失败，则继续向上抛出异常
        raise

# 仅在当前文件作为脚本直接运行时执行下面的测试代码
if __name__ == "__main__":
    try:
        llm_openai, llm_embedding = get_llm("openai")
        llm_invalid = get_llm(llm_type="invalid")

    except LLMInitializationError as e:
        # 记录致命错误，并终止程序
        logger.error(f"程序终止: {str(e)}")
