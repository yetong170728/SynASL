import os,sys
pythonpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, pythonpath)

from utils.logTool import logTool

logTool().info("test")