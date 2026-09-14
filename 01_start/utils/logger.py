import logging
#从第三方库concurrent_log_handler中导入ConcurrentRotatingFileHandler
#该处理器是RotatingFileHandle的并发安全版本，支持多进程/多线程安全写同一个日志文件并按大小滚动切分
from concurrent_log_handler import ConcurrentRotatingFileHandler

#从当前包中导入Config配置类，用于获取日志文件路径，大小和备份数据等配置
from .config import Config

class LoggerManager:
    """日志管理器类，提供统一的日志配置和获取接口"""

    #类级别的单例实例引用，确保全局只创建一个LoggerManager
    _instance = None

    #实际的logging.Logger实例引用
    _lgger = None

    def __new__(cls):
        """单例模式，确保全局只有一个日志管理器实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        #返回已存在或新创建的单例实例
        return cls._instance

    def __init__(self):
        """初始化日志管理器"""
        # 仅在首次初始化时配置日志记录器，避免重复配置
        if self._lgger is None:
            self._setup_logger()

    def _setup_logger(self):
        """配置日志记录器"""
        self._lgger = logging.getLogger(__name__)
        #设置日志级别为DEBUG，记录尽可能详细的调试和运行信息
        self._lgger.setLevel(logging.DEBUG)
        # 清空已有的日志处理器，防止重复添加导致重复输出
        self._lgger.handlers = []

        handler = ConcurrentRotatingFileHandler(Config.LOG_FILE, maxBytes=Config.MAX_BYTES, backupCount=Config.BACKUP_COUNT)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        #将配置好的处理器添加到日志记录器中
        self._lgger.addHandler(handler)

    @property
    def logger(self):
        """获取日志记录器"""
        return self._lgger

    @classmethod
    def get_logger(cls):
        """类访求，获取日志记录器实例"""
        #通过类本身创建/获取单例LoggerManager实例
        instance = cls()
        #返回内部logger，供业务代码直接使用
        return instance.logger