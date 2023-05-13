import os.path
from typing import Union, List

import fasttext
import requests
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(src: str, tgt: str):
    """

    :param src:
    :param tgt:
    :return:
    """

    # TODO handle config files

    with open("config/language_pairs.yml", "r") as stream:
        try:
            models = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

    model_name = models.get(src, {}).get(tgt, {}).get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if model_name is None:
        raise Exception(f"Language pair {src} to {tgt} not available!")
    return model, tokenizer


def _language_detection(text: List[str]) -> List[str]:
    """

    :param text:
    :return:
    """

    # TODO move to cache directory
    pretrained_lang_model = "config/lid.176.bin"
    if not os.path.exists(pretrained_lang_model):
        resp = requests.get("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
        with open(pretrained_lang_model, "wb") as f:
            f.write(resp.content)

    try:
        lang_model = fasttext.load_model(pretrained_lang_model)
    except ValueError:
        raise Exception("The fasttext language detection model is not present!")
    src = lang_model.predict(text, k=1)
    src = [lang[0].replace("__label__", "") for lang in src[0]]
    return src


def translate(text: Union[str, List[str]], *,
              src: Union[str, List[str]] = None, tgt: str) -> Union[str, List[str]]:
    """

    :param text:
    :param src:
    :param tgt:
    :return:
    """
    if isinstance(text, str):
        text = [text]

    if src is None:
        src = _language_detection(text)

    # TODO optimize grouping
    inputs = {}
    for src_lang, sentence in zip(src, text):
        sentence_list = inputs.get(src_lang, [])
        sentence_list.append(sentence)
        inputs[src_lang] = sentence_list

    output = []
    for src_lang, sentences in inputs.items():
        model, tokenizer = load_model(src_lang, tgt)

        # TODO break long sentences
        input_ids = tokenizer(sentences, padding=True, truncation=False,
                              return_attention_mask=False, return_tensors="pt").get("input_ids")
        outputs = model.generate(input_ids)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output.extend(decoded)
    return output[0] if len(output) == 1 else output
