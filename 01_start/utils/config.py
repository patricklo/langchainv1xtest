import os

class Config:
    LOG_FILE = "logfile/app.log"
    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))

    MAX_BYTES = 5*1024*1024
    #配置日志轮转时最多保留的备份文件数量，这里设置为保留3个历史日志文件
    BACKUP_COUNT = 3

    #配置使用的大模型类型：
    # - "openai"
    LLM_TYPE = "openai"
