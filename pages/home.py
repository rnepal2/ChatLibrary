import os, sys
import streamlit as st
from typing import List, Dict, Optional
from pydantic import BaseModel
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.utils.helper import get_llm, create_unique_chat_id
from src.utils.helper import get_username, update_userlog
from src.rag.graph import build_graph
from src.rag.chatbot import get_chat_chain

# constants
LOGPATH = "logs/userlog.parquet"

st.set_page_config(layout="centered", initial_sidebar_state='expanded', page_title="ChatApp")
st.markdown(open(f"{os.getcwd()}/static/style_1.txt","r").read(), unsafe_allow_html=True)
st.markdown(open(f"{os.getcwd()}/static/style_2.txt","r").read(), unsafe_allow_html=True)
st.markdown(open(f"{os.getcwd()}/static/spacing.txt","r").read(), unsafe_allow_html=True)

# Initialize session states
def initialize_sesstion_states():
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "last_chat_id" not in st.session_state:
        st.session_state.last_chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
initialize_sesstion_states()

if "username" not in st.session_state:
    st.session_state.username = get_username()
else:
    username = st.session_state.username

# chat and feedback models
class Feedback(BaseModel):
    username: str
    stars: Optional[int] = None
    comment: Optional[str] = None

class ChatMessage(BaseModel):
    chat_id: str 
    question: str
    response: str
    references: List[Dict]
    feedback: Optional[Feedback]

# chat messages and logging
messages: List[ChatMessage] = st.session_state.messages

def clear_chat_buffer():
    st.session_state.messages = list()

# SIDEBAR
with st.sidebar:
    st.subheader(":red[ChatLibrary]")
    st.write("**A generative AI powered chat app to answer your questions based on proprietary knowledge base library.**")

    with st.popover(":material/settings: :blue[Setting]", use_container_width=True):
        model = st.radio(
            label="**Model**", options=["GPT-4o-Mini", "GPT-4o", ], index=0, horizontal=False)
        temperature = st.slider("**Temperature**", min_value=0.0, max_value=2.0, step=0.1,
                                value=0.0, help="Higher value means higher creativity and low consistency in results.")
        show_ref = st.toggle("**Show References**", value=True)

    with st.popover(":material/source: :blue[Data Source]", use_container_width=True):
        st.write("*Select PMR data source(s):*")
        sources = []
        library_1 = st.checkbox("**Library_1**", value=True)
        library_2 = st.checkbox("**Library_2**", value=True)
        if library_1: sources.append("Library_1")
        if library_2: sources.append("Library_2")
        
    with st.popover(":material/info: :blue[About]", use_container_width=True):
        st.info("This is an application developed in collaboration with NPBD, CDS, and GCSO.")
        
    st.write(f":red[*Data Source:*] {'None' if len(sources) == 0 else ', '.join(sources)}")

    for i in range(5):
        st.write('')

    tokens, cost = st.columns([0.5, 0.5])
    display_expense = True
    if display_expense:
        with tokens:
            st.metric("**Total Tokens**",
                      value=st.session_state.total_tokens)
        with cost:
            st.metric(
                "**Total Cost**", value=f"${round(st.session_state.total_cost, 3)}")

    for i in range(5):
        st.write("")

    st.image(f"{os.getcwd()}/static/signin-icon.svg", width=25)
    st.write(f"Welcome :blue[{username}]")
    st.image(f"{os.getcwd()}/static/company_logo.png", width=150)

    if len(st.session_state.messages) > 0:
        clear = st.button(":material/delete: **Clear**", key="clear_conv_history",
                          on_click=clear_chat_buffer, type="secondary", help="Clears chat history.")


SYS_PROMPT = """You are an helpful assistant to answer user question based on a given documents.
                     When user question can't be answer based on provided documents, you will respond 
                     with a short message that informs the user that their question can't be answered
                     based on selected context. You can adapt your exact response based on the specific
                     user question.
                """

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYS_PROMPT), ("user", "{question}")]
)

def get_default_response(question):
    llm = get_llm(model, temperature)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question})

def update_user_feedback(stars=None, comment=None):
    _messages = []
    for chat in st.session_state.messages:
        if chat.chat_id == st.session_state.last_chat_id:
            if stars is not None: chat.feedback.stars = stars+1
            if comment is not None: chat.feedback.comment = comment
            _messages.append(chat)
        else:
            _messages.append(chat)
    st.session_state.messages = _messages

@st.dialog("Please provide any additional details")
def reason():
    with st.form('User feedback'):
        st.write(f'Why do you rate this response with selected star(s)?')
        comment = st.text_area("Comment", label_visibility="collapsed")
        submit = st.form_submit_button('Submit', type='primary')
        if submit:
            update_user_feedback(comment=comment)
            st.toast('Thank you for your feedback!', icon=":material/reviews:")

def handle_feedback(chat_id):
    feedback = st.feedback("stars", key=chat_id, on_change=reason)
    if feedback is not None:
        update_user_feedback(stars=feedback)
        
def show_reference(references):
    if len(references) > 0:
        with st.expander(":material/filter: ***References***"):
            for i, doc in enumerate(references):
                st.write(f"({i+1}) Document: {doc['filepath']}, split_id: {doc['split_id']}")

if len(sources) > 0:
    chain = build_graph()
else:
    chain = get_chat_chain(model, temperature)

def chat_pmr(human):
    config = {"configurable": {"thread_id": '2', "recursion_limit": 10}}
    response = chain.invoke({"question": human, "model": model, "collections": sources}, config=config)
    return response

def chat_bot(human):
    config = {"configurable": {"thread_id": "2", "recursion_limit": 10}}
    input_message = HumanMessage(content=human)
    response = chain.invoke({"messages": [input_message]}, config)
    print(response)
    return response

if len(sources) > 10:
    update_userlog(messages, LOGPATH)

# chatbot
human = st.chat_input('Ask me anything...')

if len(messages) == 0:
    WELCOME = "Hello! I'm an AI-powered research assistant with private knowledge base library. \
    It may take some time to complete your request. Ask me anything!"
    with st.chat_message("ai"):
        st.write(WELCOME)

for i, chat in enumerate(list(st.session_state.messages)):
    with st.chat_message(name="Question", avatar=f"{os.getcwd()}/static/chat_user_icon.png"):
        st.write(str(chat.question).replace("$", "\$"))
    with st.chat_message(name="Answer", avatar=f"{os.getcwd()}/static/chat_ai_icon.png"):
        st.write(str(chat.response).replace("$", "\$").replace('<br>', '\n'))
        if chat.chat_id == st.session_state.last_chat_id and len(sources) > 0:
            handle_feedback(chat.chat_id)
            if show_ref: show_reference(chat.references)


config = {"configurable": {"session_id": f"{username}"}}
if human is not None:
    with st.chat_message(name="Question", avatar=f"{os.getcwd()}/static/chat_user_icon.png"):
        st.write(str(human).replace("$", "\$"))

    with st.chat_message(name="Answer", avatar=f"{os.getcwd()}/static/chat_ai_icon.png"):
        with st.spinner("***Running...***"):
            with get_openai_callback() as cb:
                try:
                    if len(sources) == 0:
                        response = chat_bot(human)
                        generation = response['messages'][-1].content
                        references = []
                    else:
                        response = chat_pmr(human)
                        generation = response['generation']
                        references = [d.metadata for d in response['documents']]
                except:
                    generation = get_default_response(human)
                    references = []

                this_chat_id = create_unique_chat_id(username, human)
                st.session_state.messages.append(ChatMessage(chat_id=this_chat_id, question=human, response=generation, 
                                                 references=references, feedback=Feedback(username=st.session_state.username)))
                st.session_state.last_chat_id = this_chat_id
                st.session_state.total_tokens += cb.total_tokens
                st.session_state.total_cost += cb.total_cost
    st.rerun()
