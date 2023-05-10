from typing import Union, List

import fasttext
import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(src: str, tgt: str):
    """

    :param src:
    :param tgt:
    :return:
    """

    with open("config/config.yml", "r") as stream:
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

    pretrained_lang_model = "data/lid.176.bin"
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

    # TODO group the sentences with the same language
    output = []
    for src_lang, sentence in zip(src, text):
        model, tokenizer = load_model(src_lang, tgt)
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
        # TODO break long sentences
        outputs = model.generate(input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output.append(decoded)
    return output[0] if len(output) == 1 else output
