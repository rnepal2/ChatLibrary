import pandas as pd
import streamlit as st
import hashlib, base64
from datetime import datetime
from langchain_openai import AzureChatOpenAI
from .config import cfg, model_dict


def get_username():
    username = 'guest' # implment auth/user capture
    return username

def raise_error_and_stop(err, icon='❗'):
    st.error(err, icon=icon)
    st.stop()
    
def raise_info_and_stop(msg, icon="ℹ️"):
    st.info(msg, icon=icon)
    st.stop() 


def get_llm(model="GPT-4o-Mini", temperature=0.0):
    llm = AzureChatOpenAI(
        azure_endpoint=cfg.AZURE_OPENAI_ENDPOINT,
        api_key=cfg.AZURE_OPENAI_API_KEY,
        openai_api_version="2023-07-01-preview",
        azure_deployment=model_dict[model][0],
        model_name=model_dict[model][1],
        temperature=temperature,
    )
    return llm

def create_unique_chat_id(username: str, question: str) -> str:
    """
        Unique ID creation per news for database primary key.
    """
    link = f"{username}-{str(question).strip()[:20]}-{str(datetime.now())[:18]}"
    sha256 = hashlib.sha256()
    sha256.update(link.encode('utf-8'))
    hash_digest = sha256.digest()
    encoded = base64.b64encode(hash_digest, altchars=b'-_').decode('utf-8')
    alphanumeric = ''.join(char for char in encoded if char.isalnum()).upper()
    return alphanumeric[:10]


def update_userlog(messages, logpath):
    '''Updates the userlog data.'''
    try:
        df = pd.read_parquet(logpath)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['chat_id', 'username', 'question', 'response', 'stars', 'comment', 'timestamp'])
    existing_data = df.set_index('chat_id').to_dict('index')

    new_records = []
    for msg in messages:
        feedback_data = {
            'username': msg.feedback.username if msg.feedback else '',
            'stars': str(msg.feedback.stars) if msg.feedback and msg.feedback.stars is not None else '',
            'comment': msg.feedback.comment if msg.feedback and msg.feedback.comment is not None else ''
        }

        if msg.chat_id in existing_data:
            existing_record = existing_data[msg.chat_id]
            if (existing_record['username'] != feedback_data['username'] or
                existing_record['stars'] != feedback_data['stars'] or
                existing_record['comment'] != feedback_data['comment']):
                df.loc[df['chat_id'] == msg.chat_id, ['username', 'stars', 'comment']] = [
                    feedback_data['username'], feedback_data['stars'], feedback_data['comment']
                ]
        else:
            new_record = {
                'chat_id': msg.chat_id,
                'question': msg.question,
                'response': msg.response,
                'timestamp': str(datetime.now())[:19],
                **feedback_data
            }
            new_records.append(new_record)

    if len(new_records) > 0:
        new_df = pd.DataFrame(new_records)
        df = pd.concat([df, new_df], ignore_index=True)
    df.to_parquet(logpath, index=False)


def count_tokens(text, model_name="gpt-3.5-turbo", debug=False):
    # Using tiktoken
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model_name)
        num_tokens = len(encoding.encode(text))
        result = {"n_tokens": num_tokens, "method": "tiktoken"}
        return result
    except Exception as e:
        if debug:
            print(f"Error using tiktoken: {e}")
        pass

    # Using nltk
    try:
        import nltk
        nltk.download('punkt')
        tokens = nltk.word_tokenize(text)
        result = {"n_tokens": len(tokens), "method": "nltk"}
        return result
    except Exception as e:
        if debug:
            print(f"Error using nltk: {e}")
        pass

    tokens = text.split()
    result = {"n_tokens": len(tokens), "method": "split"}
    return result
