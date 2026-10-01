import time

import streamlit as st
from agent.react_agent import ReactAgent

#标题
st.title("智扫通机器人智能客服")
st.divider()    #分隔符

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:   #构建历史记录消息保存
    st.session_state["message"] =[]

for message in st.session_state["message"]:     #每次页面重新启动时，将message从历史记录取出并显示
    st.chat_message(message["role"]).write(message["content"])

#用户输入提示词
prompt=st.chat_input()      #内容输入框

if prompt:
    st.chat_message("user").write(prompt)       #显示用户输入的内容
    st.session_state["message"].append({"role":"user","content":prompt})

    response_messages=[]
    with st.spinner("智能客服思考中..."):
        res_stream=st.session_state["agent"].execute_stream(prompt)

        def capture(generator,cache_list):      #generator生成器 ，cache_list抓取消息为一个列表

            for chunk in generator:
                cache_list.append(chunk)

                for char in chunk:
                    time.sleep(0.01)
                    yield char

        st.chat_message("assistant").write_stream(capture(res_stream,response_messages))
        st.session_state["message"].append({"role":"assistant","content":response_messages[-1]})
        st.rerun()

