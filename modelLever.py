from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.globals import set_verbose
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import json
import os
import datetime
import base64
import uuid


model_name = "modelLever"


# 生成多模態prompt，適用於 Gemini 和 Ollama
def generatePrompt(importData):
    prompt = importData["promptData"]
    imageData = importData["imageData"]
    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{imageData}",
    }
    content_parts = []
    sys_parts = []
    text_part = {"type": "text", "text": prompt}
    content_parts.append(image_part)
    sys_parts.append(text_part)
    # SystemMessage 設定模型的上下文或行為(下prompt)
    sys_prompt_msg = SystemMessage(content=sys_parts)
    # HumanMessage 實際需要處理的圖像
    return [sys_prompt_msg, HumanMessage(content=content_parts)]


# 生成多模態prompt，適用於 OpenAI
def generateOpenAIImagePrompt():
    image_prompt_template = ImagePromptTemplate(
        input_variables=["imageData"],
        template={"url": "data:image/jpeg;base64,{imageData}"})
    sys_prompt_msg = SystemMessagePromptTemplate.from_template("{promptData}")
    promptTempwithImage = HumanMessagePromptTemplate(prompt=[image_prompt_template])
    return [sys_prompt_msg, promptTempwithImage]


# 生成多模態prompt，適用於 Gemini 和 Ollama
def generatePromptwithList(importData):
    ctxDataList = importData["txtData"]
    imageList = importData["imageData"]
    prompt = importData["promptData"]
    query = f"Answer the question only based on the information extracted from the text and images.Answer the question concisely. Question: {prompt}"
    print(f"The length of image list {len(imageList)}. The length of text list is {len(ctxDataList)}\n")
    content_parts = []
    content_parts.append({"type": "text", "text": query})
    # image_url
    for imgData in imageList:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{imgData}",
        }
        content_parts.append(image_part)
    for txtData in ctxDataList:
        text_part = {"type": "text", "text": txtData}
        content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

# 生成多模態prompt，適用於 OpenAI
def generateOpenAIPromptwithList(importData):
    prompt = importData["promptData"]
    ctxDataList = importData["txtData"]
    imageData = importData["imageData"]

    msgContent = []

    query = f"Answer the question only based on the information extracted from the text, images and tables.Answer the question concisely. Question: {prompt}"
    # 將問題添加到消息內容中
    msgContent.append({"type": "text", "text": query})
    # 將圖片添加到消息內容中
    for img in imageData:
        msgContent.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        )
    # 將文本添加到消息內容中
    for txtData in ctxDataList:
        msgContent.append({"type": "text", "text": txtData})
    
    # 返回消息內容
    return [HumanMessage(content=msgContent)]



# 創建LLM模型
def createModel(modelService , model , temp):
    # The temperature by default Google 0.7, OpenAI 0.7, Ollama 0.8. For summary pipeline, I set them all to 0.8. But for the Q&A pipeline, you can set up by dragging the bar
    llm = {}

    modelSel = model
    if modelService == "OpenAI":
        llm = ChatOpenAI(model=modelSel , temperature = temp)
    elif modelService == "Google Gemini":
        llm = ChatGoogleGenerativeAI(model=modelSel , temperature = temp)
    elif modelService == "Ollama":
        llm = ChatOllama(model=modelSel , temperature = temp)
    return llm



# 總結表格、文字與圖像
def summarizeDatafromPDF(extractData):
    # Prompt模板
    prompt = """You are an assistant tasked with summarizing tables, text and images. Summarize the content from table, 
                text and image chunks. Pay attention to the term definition, time period, numbers, list, all the key points, etc.  
                Table or text content are : {dataContent}"""
    # 利用 ChatPromptTemplate 建立一個可重複使用的模板，將內容（表格、文字）傳遞到 {dataContent} 中。
    promptTemplate = ChatPromptTemplate.from_template(prompt)
    # 生成總結內容時將temperature設定為0.8
    llm = createModel(st.session_state.summaryService , st.session_state.summaryModelSel , 0.8)

    # 建chain 1.建立一個初始數據映射，將輸入數據包裝成字典格式 2.利用之前建立的 Prompt 模板，將數據插入模板中
    # 3.使用大語言模型根據 Prompt 生成結果 4.使用 StrOutputParser() 將結果轉換為字串
    summarizeChain = {"dataContent": lambda x: x} | promptTemplate | llm | StrOutputParser()
    # print(type(extractData["textElements"]))
    tableSummaries = []
    textSummaries = []
    for tbl in extractData["tableElements"]:
        print(f"here's the table {tbl}\n")
        response = summarizeChain.invoke(tbl)
        print(f"here's the table summary {response}\n")
        tableSummaries.append(response)
    for txt in extractData["textElements"]:
        # print(f"here's the text {txt}\n")
        response = summarizeChain.invoke(txt)
        textSummaries.append(response)
    imageSummaries = []
    for img in extractData["imgPath"]:
        # 將圖片編碼為 Base64，便於傳遞到模型進行處理
        imageBase64 = encodeImageBase64(img)
        chain = generatePrompt | llm | StrOutputParser()
        # 將圖片傳給模型，open ai 和 ollama跟gemini的處理方式不同
        if st.session_state.summaryService == "OpenAI":
            promptTempwithImage = generateOpenAIImagePrompt()
            chat_prompt_template = ChatPromptTemplate.from_messages(promptTempwithImage)
            chain = chat_prompt_template | llm | StrOutputParser()
        response = chain.invoke({"imageData": imageBase64, "promptData": "Please describe the image and summarize the content concisely"})
        # print(response)
        imageSummaries.append(response)
    print(f"The size of text summary is {len(textSummaries)}\n The size of table summary is {len(tableSummaries)}\n The size of image summary is  {len(imageSummaries)}\n")
    return {"textSummaries": {"mediatype": "text", "payload": extractData["textElements"], "summary": textSummaries},
            "tableSummaries": {"mediatype": "text", "payload": extractData["tableElements"], "summary": tableSummaries},
            "imageSummaries": {"mediatype": "image", "payload": extractData["imgPath"], "summary": imageSummaries}}


# 創建retriever
def retrieverGenerator(summarizedData):
    # 創建vectore DB，存總結內容  # 使用OpenAI的embeddings
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())           
    # 創建內存空間，存原始數據
    store = InMemoryStore()
    # 創建id_key唯一識別碼，之後用來建關聯
    id_key = "rec_id"

    # 創建retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )

    # Extract the structured data from former function
    for key in summarizedData.keys():
        mediaType = summarizedData[key]["mediatype"]
        summary = summarizedData[key]["summary"]
        payload = summarizedData[key]["payload"]
        print(f"size of mediatype {len(mediaType)}. The size of summary {len(summary)}. The size of payload {len(payload)}")
        # uuid4 產生隨機的 UUID 唯一識別碼
        docs_ids = [str(uuid.uuid4()) for _ in summary]

        # 避免空列表導致程式停止
        if (len(summary) == 0):
            continue
        if (mediaType == "text"):
           summaryDoc = [
               Document(page_content=s, metadata={id_key: docs_ids[i], "mediaType": mediaType})
                for i, s in enumerate(summary)
            ]
        elif (mediaType == "image"):
           summaryDoc = [
               Document(page_content=s, metadata={id_key: docs_ids[i], "mediaType": mediaType, "source": payload[i]})
                for i, s in enumerate(summary)
            ]

        # 將總結內容添加到vectore store，將原始數據添加到內存空間
        retriever.vectorstore.add_documents(summaryDoc)
        retriever.docstore.mset(list(zip(docs_ids, payload)))

    # 將retriever存入session，供其他函數調用
    st.session_state.vectorretriever = retriever


def askLLM(query):
    # 從session中獲取retriever
    retriever = st.session_state.vectorretriever
    # 使用retriever將問題向量化後送至向量資料庫進行相似性搜索
    searchDocs = retriever.vectorstore.similarity_search(query)
    #print(f"This is the vector search result {searchDocs[0]}\n ")
    imageData = []
    txtData = []
    # 使用HTML顯示
    relevantImages = "<br /><br />  <h2>Below are the relevant images retrieved</h2>"
    for doc in searchDocs:
        rec_id = doc.metadata["rec_id"]
        mediaType = doc.metadata["mediaType"]
        # 透過rec_id從docstore中獲取對應的原始數據
        ctxContent = retriever.docstore.mget([rec_id])
        print(f"This is the record content {rec_id}\n   {ctxContent}\n")
        if(mediaType == "text"):
            txtData.append(ctxContent[0])
        elif(mediaType == "image"):
            # 將圖片編碼為Base64，便於傳遞到模型進行處理
            imgB64Enc = encodeImageBase64(ctxContent[0])
            imageData.append(imgB64Enc)
            # 在 HTML 中嵌入圖片數據
            relevantImages = relevantImages + f"<br /><br /><img   width=\"60%\" height=\"30%\"  src=\"data:image/jpeg;base64,{imgB64Enc}\">  "
    llmModel = createModel(st.session_state.serviceSel , st.session_state.modelSel , st.session_state.tempSel)
    chain = {}
    modelService = st.session_state.serviceSel

    if (modelService == "OpenAI"):
        chain = StrOutputParser()(llmModel(generatePromptwithList()))

    else:
        chain = generatePromptwithList | llmModel | StrOutputParser()
    response = chain.invoke({"imageData": imageData, "txtData": txtData , "promptData": query})
    if (len(imageData) == 0):
        relevantImages = ""
    return response + relevantImages

# 將圖片編碼為Base64 Base64編碼允許將二進制圖片數據轉換為文本格式
def encodeImageBase64(imgPath):
    with open(imgPath, "rb") as imgContent:
        base64Data = base64.b64encode(imgContent.read())
        return base64Data.decode("utf-8")
