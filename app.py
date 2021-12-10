import streamlit as st

import io
import os
import yaml

from predict import load_tokenizer, load_reader, load_retriever, get_prediction, get_text

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'

def main():
    st.title("Ask Anything!")
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    st.text("Fetching Model...")
    tokenizer = load_tokenizer(config)
    model = load_reader(config)
    retriever = load_retriever(config, tokenizer)

    model.eval()
    st.text("Model Fetched!")

    question = st.text_input('무엇이든 물어보세요')
    print(question)

    if question:
        print('Fetching Answer...')
        st.write("Fetching Answer...")
        pred = get_prediction(tokenizer, model, retriever, question, config)
        prediction = pred[0]['prediction_text']
        answer = prediction['answer']
        context = prediction['context']
        start_logit = prediction["start_logit"]
        end_logit = prediction["end_logit"]

        st.write(f'Answer is {answer}')
        st.write(get_text(context, start_logit, end_logit))
        st.write("Passage Retrieval은 미완성 기능입니다!")


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')