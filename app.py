import streamlit as st

import io
import os
import yaml

from predict import load_tokenizer, load_reader, load_retriever, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = 'password'

def main():
    st.title("Ask Anything!")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    tokenizer = load_tokenizer(config)
    model = load_reader(config)
    retriever = load_retriever(config, tokenizer)

    model.eval()

    question = st.text_input('무엇이든 물어보세요')

    if question:
        get_prediction(model, question)
        st.write(f'')

    if question:
        st.write("Fetching Answer...")
        _, pred = get_prediction(tokenizer, model, retriever, question)[0]
        answer = pred['answer']

        st.write(f'Answer is {answer}')


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