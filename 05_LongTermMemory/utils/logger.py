import logging

from concurrent_log_handler import ConcurrentRotatingFileHandler
from .config import Config

class LoggerManager:
    _instance = None
    _logger= None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers = []

        handler = ConcurrentRotatingFileHandler(
            Config.LOG_FILE,
            maxBytes=Config.MAX_BYTES,
            backupCount=Config.BACKUP_COUNT,
        )

        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        self._logger.addHandler(handler)

    @property
    def logger(self):
        return self._logger

    @classmethod
    def get_logger(cls):
        instance = cls()
        return instance.logger