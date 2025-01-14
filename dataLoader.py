
from PIL import Image
import fitz
import io
import modelLever

module_name = "dataLoader"

# 將表格數據轉換成字串格式
def turnTbl2Str(tblData):
    tblStr  = "\n".join([" | ".join(map(str, row)) for row in tblData])
    return tblStr

# 從 PDF文件中提取元素，並將其分類為文字、表格和圖片
def ExtractDataFromPDF(pdfFileContent):
    # 使用 PyMuPDF (fitz) 開啟 PDF 文件，使用io.BytesIO()將二進制內容轉換為一個類文件對象 
    pdfLoader = fitz.open("pdf", io.BytesIO(pdfFileContent))

    textElements = []
    tableElements = []
    imgPath = []
    # 逐頁處理 
    for pageIndex in range(len(pdfLoader)):
        pageContent = pdfLoader[pageIndex]

        # 翠取文字
        text = pageContent.get_text()
        textElements.append(text)

        # 翠取表格 設定表格識別策略，水平策略為線(水平線確定表格列(row)的邊界)，垂直策略為文字排列(垂直對齊來判斷行(column)的位置)
        tables = pageContent.find_tables(horizontal_strategy="lines", vertical_strategy="text")
        # 逐個表格處理
        for table in tables:
            # 將表格轉換成字串格式
            tableElements.append(turnTbl2Str(table.extract()))

        # 翠取圖片
        for imgIndex, imgData in enumerate(pageContent.get_images(), start=1):
            # 每個圖像都由一個引用編號 (xref) 唯一標識，用來從 PDF 中提取圖像的關鍵
            xref = imgData[0]
            imgData = pdfLoader.extract_image(xref)
            imgByte = imgData["image"]
            # png, jpeg.....
            imgExt = imgData["ext"]
            imgContent = Image.open(io.BytesIO(imgByte))
            # 保存圖像
            imgContent.save(open(f"./pdfimages/pdfImage{pageIndex + 1}_{imgIndex}.{imgExt}", "wb"))
            # 紀錄圖像路徑
            imgPath.append(f"./pdfimages/pdfImage{pageIndex + 1}_{imgIndex}.{imgExt}")
    print(f"{len(tableElements)} Tables\n{len(textElements)} Text Passages\n{len(imgPath)} Images")
    return {"textElements": textElements, "tableElements": tableElements, "imgPath": imgPath}


# 處理PDF文件
def processData(pdfFile):
    print("Extract Data")
    # pdfFile.getvalue() > 提取檔案內容(形式為二進制資料類型)  # 提取PDF文件中的文字、表格和圖片
    extractData = ExtractDataFromPDF(pdfFile.getvalue())
    # 根據從PDF文件中萃取出的文字、表格和圖片，生成總結 # modelLever.py的summarizeDatafromPDF函數
    summarizedData = modelLever.summarizeDatafromPDF(extractData)
    # 創建retriever # modelLever.py的retrieverGenerator函數
    modelLever.retrieverGenerator(summarizedData)

