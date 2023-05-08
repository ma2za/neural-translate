import yaml
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(src: str, dst: str):
    """

    :param src:
    :param dst:
    :return:
    """

    with open("config/config.yml", "r") as stream:
        try:
            models = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise exc

    model_name = models.get(src, {}).get(dst, {}).get("model")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if model_name is None:
        raise Exception(f"Language pair {src} to {dst} not available!")
    return model, tokenizer


def translate(sentence: str, src: str, dst: str) -> str:
    """

    :param sentence:
    :param src:
    :param dst:
    :return:
    """

    model, tokenizer = load_model(src, dst)
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(input_ids)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded
