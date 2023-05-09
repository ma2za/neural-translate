import yaml
from langdetect import detect as detect_language
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


def translate(sentence: str, *, src: str = None, tgt: str) -> str:
    """

    :param sentence:
    :param src:
    :param tgt:
    :return:
    """

    if src is None:
        src = detect_language(sentence)
    model, tokenizer = load_model(src, tgt)
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded
