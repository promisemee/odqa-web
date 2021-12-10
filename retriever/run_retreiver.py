import pandas as pd
from .bm25 import SparseRetrieval_BM25
from datasets import Dataset, DatasetDict
from typing import Callable, List
from transformers import TrainingArguments

def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    question: str,
    training_args: TrainingArguments,
    configs,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents_cleaned.json",
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval_BM25(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_tokenized()
    
    _, context_list = retriever.retrieve(question, topk=configs["top_k_retrieval"])

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        df = pd.DataFrame({
                "context": [" ### ".join(
                    [context for context in context_list]
                )],
                "question": [question],
                "id":[1]
            })

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    
    datasets = DatasetDict({"validation": Dataset.from_pandas(df)})
    print(datasets)
    return datasets