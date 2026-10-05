from mcp.server.lowlevel import Server
from mcp.types import Resource,Tool,TextContent
from utils.config import Config
from mix_text_search import MilvusSearchManager
from utils.logger import LoggerManager

logger = LoggerManager.get_logger()
mcp = Server("rag_mcp_server")

@mcp.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Listingg tools.....")
    return [
        Tool(
            name="search_documents",
            description="执行文档搜索",
            inputSchema={
                # 指定输入类型为对象
                "type": "object",
                # 定义对象的各个属性
                "properties": {
                    # 定义query_text属性，存储搜索查询文本
                    "query_text": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置属性的描述信息
                        "description": "执行搜索的内容"
                    },
                    # 定义filter_query属性，存储过滤条件的自然语言描述
                    "filter_query": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置默认值为"##None##"
                        "default": "##None##",
                        # 设置属性的描述信息，包含默认值和示例说明
                        "description": "过滤条件的自然语言描述内容,默认值为##None##。如:文章发布时间在2025年9月3号到5号之间的文章,作者是新智元的文档"
                    },
                    # 定义search_type属性，存储搜索类型
                    "search_type": {
                        # 指定属性类型为字符串
                        "type": "string",
                        # 设置默认值为"hybrid"
                        "default": "hybrid",
                        # 设置属性的描述信息，说明可选值和默认值
                        "description": "可选 dense、sparse、hybrid，其中dense为语义搜索、sparse为全文搜索或关键词搜索、hybrid为混合搜索，默认为hybrid"
                    },
                    # 定义limit属性，存储返回结果的数量限制
                    "limit": {
                        # 指定属性类型为数字
                        "type": "number",
                        # 设置默认值为2
                        "default": 2,
                        # 设置属性的描述信息，说明默认值
                        "description": "结果返回的数量,默认值为2"
                        # "description": "Number of results,default 2"
                    }
                },
                # 列出输入对象的必需属性
                # 指定必需的属性列表
                "required": ["query_text", "filter_query", "search_type", "limit"]
            }
        )
    ]

# 声明 call_tool 函数为一个工具调用的接口
@mcp.call_tool()
# 返回值： TextContent对象的列表
async def call_tool(name: str, arguments: dict) -> list[TextContent]:

    if name != "search_documents":
        raise ValueError(f"unknown tool: {name}")
    query_text = arguments["query_text"]
    search_type = arguments["search_type"]
    limit = arguments["limit"]
    filter_query = arguments["filter_query"]
    # 验证query_text参数是否存在
    if not query_text:
        # 如果不存在，抛出值错误异常
        raise ValueError("Query is required")
    # 验证filter_query参数是否存在
    if not filter_query:
        # 如果不存在，抛出值错误异常
        raise ValueError("filter_query is required")
    # 验证search_type参数是否存在
    if not search_type:
        # 如果不存在，抛出值错误异常
        raise ValueError("Search type is required")
    # 验证limit参数是否存在
    if not limit:
        # 如果不存在，抛出值错误异常
        raise ValueError("Limit is required")
    try:
        # 创建MilvusSearchManager实例，指定Milvus服务器地址和数据库名称
        search_manager = MilvusSearchManager(
            milvus_uri= Config.MILVUS_URI,
            db_name= Config.MILVUS_DB_NAME
        )

        # 执行混合 搜索示例
        filter_result = search_manager.search_with_filter(
            collection_name=Config.MILVUS_COLLECTION_NAME,
            query_text=query_text,
            filter_query=filter_query,
            search_type=search_type,
            limit=limit
        )
        if filter_result["success"]:
            # 打印搜索成功的信息
            logger.info(f"过滤搜索成功,共搜索到{filter_result['total_results']}条")
            if filter_result["results"] and len(filter_result["results"]) > 0:
                filtered_items = [
                    (
                        res.entity.get("title", ""),
                        res.entity.get("content_chunk", ""),
                        res.entity.get("link", ""),
                        res.entity.get("pubAuthor", ""),
                        res.entity.get("pubDate", ""),
                        res.distance
                    ) for res in filter_result["results"][0]
                ]
                filtered_result_string = ""
                # 遍历所有结果项，索引从1开始
                for idx, item in enumerate(filtered_items, 1):
                    # 解包元组，获取各个字段的值
                    title, content_chunk, link, pubAuthor, pubDate, distance = item
                    # 构建单条记录的字符串
                    record = (
                        f"文章标题: {title}\n"
                        f"文章原始链接: {link}\n"
                        f"文章发布者: {pubAuthor}\n"
                        f"文章发布时间: {pubDate}\n"
                        f"文章内容片段: {content_chunk}\n\n\n"
                    )
                    # 将记录追加到结果字符串
                    filtered_result_string += record
                # 打印完整的搜索结果字符串
                logger.info(f"过滤搜索结果:\n{filtered_result_string}")
                # 返回一个包含查询结果的 TextContent 对象
                # 创建TextContent对象，将搜索结果封装为文本内容并返回
                return [TextContent(type="text", text=filtered_result_string)]
            # 如果搜索失败
        else:
            # 打印搜索失败的错误信息
            logger.error(f"过滤搜索失败: {filter_result['error']}")
            # 检查是否有建议查询
            if "suggestions" in filter_result:
                # 打印建议查询信息
                logger.info(f"   建议查询: {filter_result['suggestions']}")
            # 返回一个包含查询结果的 TextContent 对象
            # 返回未检索到结果的提示信息
            return [TextContent(type="text", text="\n未检索到相关结果")]
    except Exception as e:
        # 记录主程序执行异常的错误日志
        logger.error(f"主程序执行异常: {e}")
        logger.info("程序异常终止。")
        # 返回程序异常的错误信息
        return [TextContent(type="text", text="\n主程序执行异常，程序异常终止")]

if __name__ == "__main__":
    # 初始化并运行服务器，使用streamable_http传输协议
    mcp.run(transport="streamable_http")

