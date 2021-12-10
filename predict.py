import streamlit as st

import pandas as pd

from datasets import (
    Dataset,
    DatasetDict,
)

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from reader import run_mrc
from model import CustomModel
from retriever import SparseRetrieval_BM25

def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_name"]
        if config["tokenizer_name"]
        else config["reader_path"],
        use_fast=True,
    )

    return tokenizer

def load_reader(config):
    model_config = AutoConfig.from_pretrained(
        config["config_name"]
        if config["config_name"]
        else config["reader_path"]
    )
    
    model = CustomModel.from_pretrained(
        config["reader_path"],
        from_tf=bool(".ckpt" in config["reader_path"]),
        config=model_config
    )

    return model

def load_retriever(config, tokenizer):
    print(config["data_path"], config["context_path"])
    retriever = SparseRetrieval_BM25(
        tokenize_fn=tokenizer, data_path=config["data_path"], context_path=config["context_path"]
    )
    retriever.get_tokenized()

    return retriever



def get_prediction(tokenizer, reader, retriever, question, config):
    # retrieve passage
    _, context_list = retriever.retrieve(question, topk=config.top_k_retrieval)

    df = pd.DataFrame({
                "context": [" ".join(
                    [context for context in context_list]
                )],
                "question": [question],
                "id":[1]
            })

    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})

    # read passage
    parser = HfArgumentParser(TrainingArguments)
    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    prediction = run_mrc(config, training_args, datasets, tokenizer, reader)

    return prediction