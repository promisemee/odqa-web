    # coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-processing
Post-processing utilities for question answering.
"""
import collections
import os
import math
import re
from typing import Optional, Tuple, Any

import numpy as np
from tqdm.auto import tqdm

import torch
import random
from transformers import is_torch_available, PreTrainedTokenizerFast, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from datasets import DatasetDict, Dataset

# for data-preprocessing
import pandas as pd

def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()


    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []
        current_predictions = {}

        # 현재 example에 대한 모든 feature 생성합니다.
        for feature_index in feature_indices:
            # 각 featureure에 대한 모든 prediction을 가져옵니다.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            # minimum null prediction을 추가합니다.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # offset을 사용하여 original context에서 answer text를 수집합니다.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 예측값에 확률을 포함합니다.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
            pred["final_score"] = prob * 1 / (math.sqrt(len(pred["text"])))

        # predictions.sort(key = lambda x:x["final_score"], reverse=True)

        # best prediction을 선택합니다.
        if not version_2_with_negative:

            idx = 0
            raw_data = predictions[0]["text"]

            # 빈답안 제거
            while(raw_data == "." and idx <n_best_size):
                raw_data = predictions[idx]["text"]
                idx += 1

            # post-preprocess, # 제거
            post_data = re.sub('\S*#\S*', '', raw_data)
            current_predictions["answer"] = post_data.strip()
        else:
            # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold를 사용해서 null prediction을 비교합니다.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]]["answer"] = ""
            else:
                all_predictions[example["id"]]["answer"] = best_non_null_pred["text"]

        current_predictions["context"] = context
        current_predictions["start_logit"] = offsets[0]
        current_predictions["end_logit"] = offsets[1]

        all_predictions[example["id"]] = current_predictions

        # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    return all_predictions


def check_no_error(
    configs,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # last checkpoint 찾기.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    # Tokenizer check: 해당 script는 Fast tokenizer를 필요로합니다.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    max_seq_length = min(configs["max_seq_length"], tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length

def clean_data(text):
    # 데이터 전처리
    text = re.sub('\|', ' ', text)
    text = re.sub('p=[0-9 ]*', '', text)
    text = re.sub('링크=[\S]*', '', text)
    text = re.sub('http\S+', '', text)
    text = re.sub('[0-9가-힣a-zA-Z]*=[0-9가-힣a-zA-Z–§]*', '', text)
    text = re.sub('[;#*]', '', text)
    text = re.sub('\\\\n', '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\s+', ' ', text)

    return text