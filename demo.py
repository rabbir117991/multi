import drawUI as du
import os
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # drawUI.py的initSession() # 初始化session
    du.initSession()
    # 創建資料夾儲存圖片
    imgFolder = "./pdfimages"
    if not os.path.exists(imgFolder):
        os.makedirs(imgFolder)
    # 讀取mmconf.json
    du.readConf("./mmconf.json")
    # 使用streamlit繪製UI
    du.drawUI("Multimodal RAG")
