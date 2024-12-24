import sys

class CustomeException(Exception):
    def __init__(self, errorMessage, errorDetails: sys):
        self.errorMessage = errorMessage
        _,_,exc_tb = errorDetails.exc_info()

        print(exc_tb)

        self.lineNo = exc_tb.tb_lineno
        self.fileName = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occured in python script name [{0}] Line Number [{1}] Error Message [{2}]".format(
        self.fileName, self.lineNo, str(self.errorMessage))
    
if __name__=="__main__":
    try:
        a=1/0

    except Exception as e:
        #print(e)
        raise CustomeException(e,sys)
