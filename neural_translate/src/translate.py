def load_model(src: str, dst: str):
    """

    :param src:
    :param dst:
    :return:
    """
    
    if src == "en" and dst == "de":
        from transformers import FSMTForConditionalGeneration, FSMTTokenizer
        mname = "facebook/wmt19-en-de"
        tokenizer = FSMTTokenizer.from_pretrained(mname)
        model = FSMTForConditionalGeneration.from_pretrained(mname)
    else:
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
