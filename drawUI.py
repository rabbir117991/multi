
import streamlit as st
import json
import time
import dataLoader
import modelLever

module_name = "drawUI"

# 讀取mmconf.json
def readConf(confPath):
    with open(confPath, "r") as confFile:
        confData = json.load(confFile)
    st.session_state.confData = confData
    ragModel = [key for key in confData["ragModel"].keys()]
    summaryModelServices = [key for key in confData["summaryModel"].keys()]
    st.session_state.apiServiceList = ragModel
    st.session_state.summaryModelService = summaryModelServices


def onChooseSummaryService():
    st.session_state.summaryModelDisabled = False
    st.session_state.uploaderDisabled = False
    st.session_state.summaryModelSelOptions = st.session_state.confData["summaryModel"][st.session_state.summaryService]


def serviceSelect():
    confData = st.session_state.confData
    if ("serviceSel" not in st.session_state) or bool(st.session_state["serviceSel"]):
        st.session_state.modelSelDisabled = False
        st.session_state.modelSelOptions = confData["ragModel"][st.session_state["serviceSel"]]
    else:
        st.session_state.modelSelOptions = []
        st.session_state.modelSelDisabled = True


# 初始化session
def initSession():
    # Control the usability of model selection
    if "modelSelDisabled" not in st.session_state:
        st.session_state.modelSelDisabled = True
    # Save the model selection to session state after selecting
    if "modelSelOptions" not in st.session_state:
        st.session_state.modelSelOptions = []
    # Initialize the llmAgent storage
    if "llmAgent" not in st.session_state:
        st.session_state.llmAgent = {}
    # Log the info to be presented
    if "serviceInfo" not in st.session_state:
        st.session_state.serviceInfo = ""
    # Log the message history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Store the dataframe of uploaded report
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = {}
    if "apiServiceList" not in st.session_state:
        st.session_state.apiServiceList = []
    # Store the conf file
    if "confData" not in st.session_state:
        st.session_state.confData = {}
    # Store the vision model to parse data
    if "summaryModelService" not in st.session_state:
        st.session_state.summaryModelService = []
    # Usability of file uploader
    if "uploaderDisabled" not in st.session_state:
        st.session_state.uploaderDisabled = True
    # Usability of summary model selection
    if "summaryModelDisabled" not in st.session_state:
        st.session_state.summaryModelDisabled = True
    # Options of summary vision models
    if "summaryModelSelOptions" not in st.session_state:
        st.session_state.summaryModelSelOptions = []
    # Add a blob to store the retriever
    if "vectorretriever" not in st.session_state:
        st.session_state.vectorretriever = {}
    # Time elapsed when processing pdf file
    if "timeelapsed" not in st.session_state:
        st.session_state.timeelapsed = 0


def drawUI(title):
    st.title(title)
    #Initialize the chat window
    # 顯示歷史聊天記錄
    for message in st.session_state.messages:
        # 角色
        with st.chat_message(message["role"]):
            # 訊息內容 # unsafe_allow_html 允許回應中包含 HTML
            st.write(message["content"] , unsafe_allow_html = True)
    #Create configuration side menu
    # 設定側邊欄
    with st.sidebar:
        # 選擇API服務和模型生成總結 # radio單選按鈕 
        st.radio(
            "Choose the API service",
            options=st.session_state.summaryModelService,
            # 存到st.session_state.summaryService
            key = "summaryService",
            # 選擇API服務後，呼叫onChooseSummaryService函數
            on_change = onChooseSummaryService,
            # 預設選項
            index = None
        )
        # selectbox下拉式選單
        st.selectbox(
            label="Please choose model of API service",
            options=st.session_state.summaryModelSelOptions,
            key="summaryModelSel",
            placeholder="Choose the model",
            disabled=st.session_state.summaryModelDisabled,
            index=None
        )
        # 上傳PDF文件
        pdfFile = st.file_uploader("Upload your pdf file from here" , key = "fileuploader", type=["pdf"], disabled=st.session_state.uploaderDisabled)
        # 花費時間
        elapsedTime = 0
        if (pdfFile is not None) and (st.session_state.summaryModelSel is not None) and (st.session_state.uploaderDisabled == False):
            # 開始時間
            startTime = time.perf_counter()
            with st.spinner("Processing PDF..."):
                # dataLoader.py的processData函數 # 處理PDF文件
                dataLoader.processData(pdfFile)
                st.session_state.uploaderDisabled = True
            # 結束時間
            endTime = time.perf_counter()
            elapsedTime = endTime - startTime
            st.session_state.timeelapsed = elapsedTime 
        st.info(f"The time cost {round(st.session_state.timeelapsed, 2)}s" , icon= "⏲️")
        st.divider()

        # 選擇API服務和模型生成回應
        st.selectbox(
            label="Choose the API service",
            options=st.session_state.apiServiceList,
            key="serviceSel",
            index=None,
            on_change = serviceSelect,
            placeholder="Choose the API service"
        )
        st.selectbox(
            label="Please choose model of API service",
            options=st.session_state.modelSelOptions,
            disabled=st.session_state.modelSelDisabled,
            key="modelSel",
            placeholder="Choose the model"
        )
        # select_slider 滑桿
        st.select_slider(
            label="Select the temperature",
            options=[0, 0.2, 0.5, 1],
            disabled=st.session_state.modelSelDisabled,
            key="tempSel"
        )
    # := 同時進行賦值和條件判斷 # st.chat_input() 創建一個聊天輸入框
    if prompt := st.chat_input("Please share what do you want to know..."):
        # 將使用者輸入的prompt添加到聊天記錄中
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                response = modelLever.askLLM(prompt)
                # unsafe_allow_html 允許回應中包含 HTML
                st.write(response , unsafe_allow_html = True)
        st.session_state.messages.append({"role": "assistant", "content": response})

