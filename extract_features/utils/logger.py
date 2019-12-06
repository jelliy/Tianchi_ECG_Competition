import sys
from datetime import datetime

from utils import system


class Tee(object):
    __message__ = ""

    def __init__(self, filename, file_log_enabled = True):
        self.terminal = sys.stdout
        self.file_log_enabled = file_log_enabled
        if file_log_enabled:
            self.logfile = open(filename, "w")

    def write(self, message):
        self.__message__ += message
        if '\n' in self.__message__:
            if len(self.__message__.strip()) > 0:
                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                self.__message__ = current_time + " - " + self.__message__

            self.terminal.write(self.__message__)
            if self.file_log_enabled:
                self.logfile.write(self.__message__)
                self.logfile.flush()
            self.__message__ = ""

    def flush(self):
        self.terminal.flush()
        if self.file_log_enabled:
            self.logfile.flush()


def enable_logging(prefix="log", file_log_enabled = True):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    system.mkdir('logs')
    sys.stdout = Tee('logs/' + prefix + '_' + current_time + '.log', file_log_enabled)
