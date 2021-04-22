from logbook import Logger, TimedRotatingFileHandler
import logbook
import os
import time

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(FILE_DIR, '../log/')
logbook.set_datetime_format('local')


def log_type(record, handler):
    log_info = "[{date}] [{level}] [{filename}] [{func_name}] [{lineno}]\n{msg}".format(
        date=record.time,# 日志时间
        level=record.level_name,                       # 日志等级
        filename=os.path.split(record.filename)[-1],   # 文件名
        func_name=record.func_name,                    # 函数名
        lineno=record.lineno,                          # 行号
        msg=record.message                             # 日志内容
    )
    return log_info


class LogHelper(object):
    def __init__(self, log_dir=BASE_DIR, log_name='log.log', backup_count=10, log_type=log_type):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        self.log_dir = log_dir
        self.backup_count = backup_count
        handler = TimedRotatingFileHandler(filename= os.path.join(self.log_dir, log_name),
                                           date_format='%Y-%m-%d',
                                           backup_count=self.backup_count)
        self.handler = handler
        if log_type is not None:
            handler.formatter = log_type
        handler.push_application()

    def get_current_handler(self):
        return self.handler

    @staticmethod
    def make_logger(level, name=str(os.getpid())):
        return Logger(name=name, level=level)

def log_helper(log_file, content):
    log_file.write( f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, {content}\n" )
    return 

if __name__ == "__main__":
    g_log_helper = LogHelper(log_name='panos.log')
    log = g_log_helper.make_logger(level=logbook.INFO)
    log.critical("critical")
    log.error("error")
    log.warning("warning")
    log.notice("notice")
    log.info("info")
    log.debug("debug")
    pass


