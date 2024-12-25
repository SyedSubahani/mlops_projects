import logging
import os
from datetime import datetime

logFile=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

logPath=os.path.join(os.getcwd(),"logs")

os.makedirs(logPath,exist_ok=True)

logFilePath=os.path.join(logPath,logFile)


logging.basicConfig(level=logging.INFO, 
                    filename=logFilePath,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
                    
)
