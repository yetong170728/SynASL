from logging import Logger, Formatter, FileHandler, StreamHandler
from logging import INFO, DEBUG
from logging import Logger
from pathlib import Path
from datetime import datetime

class logTool(Logger):
    def __init__(self, name=None, 
                 slevel=INFO, flevel=None,
    fformat=Formatter("Time：%(asctime)s - Message(Level:%(levelname)s)[%(funcName)s]-Line:%(lineno)d: %(message)s "),
                 sformat=None,
                 use_terminal=True,
                 use_file=True, log_path=Path.cwd()):
        if name is None:
            now = datetime.now()
            try:file_name = Path(__file__).stem # for *.py files
            except: file_name = sys.argv[0]
            name = "log_" + file_name +f"_{now.month:02d}{now.day:02d}_{now.hour:02d}{now.minute:02d}{now.second:02d}"
        log_name = log_path.joinpath(name +'.log')
        
        super(logTool, self).__init__(name=name)
        if sformat is None:
            sformat = fformat
        if flevel is None:
            flevel = slevel
        if use_file:
            self._setUpFhandler(log_path.joinpath(log_name), fformat, flevel)
        if use_terminal:
            self._setUpShandler(sformat,slevel)
        self.log_name = log_name
    
    def _setUpFhandler(self, file_name,format,level):
        fhandler = FileHandler(file_name)
        fhandler.setFormatter(format)
        fhandler.setLevel(level)
        self.addHandler(fhandler)
        
    def _setUpShandler(self, format,level):
        shandler = StreamHandler()
        shandler.setFormatter(format)
        shandler.setLevel(level)
        self.addHandler(shandler)

if __name__ == "__main__":
    logTool().info("test loggin info")