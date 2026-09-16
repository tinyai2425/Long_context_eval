# OMC 用例：Ceval 顶层字段 + sentences[0].prompt（已套 chat template）

from transformers import AutoTokenizer

mp = None
_tokenizer_cache = {}


def set_model_config(model_module):
    global mp
    mp = model_module


def _get_tokenizer():
    path = mp.TOKENIZER_CONFIG_PATH
    if path not in _tokenizer_cache:
        _tokenizer_cache[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _tokenizer_cache[path]


def enable_thinking_flag():
    if mp is None:
        raise ValueError("You must call set_model_config(mp) first.")
    return bool(getattr(mp, "ENABLE_THINKING", False))


def render_chat_template_messages(messages, enable_thinking=None):
    tokenizer = _get_tokenizer()
    if enable_thinking is None:
        enable_thinking = enable_thinking_flag()
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def count_chat_tokens(messages, enable_thinking=None):
    tokenizer = _get_tokenizer()
    if enable_thinking is None:
        enable_thinking = enable_thinking_flag()
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        ids = tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return len(ids)


def _common_top_level():
    if mp is None:
        raise ValueError("You must call set_model_config(mp) first.")
    return {
        "inferType": mp.INFER_TYPE,
        "tokenizerType": mp.TOKENIZER_TYPE,
        "tokenizerPath": mp.TOKENIZER_PATH,
        "modelType": mp.MODEL_TYPE,
        "modelPath": mp.MODEL_PATH,
        "weightDir": mp.WEIGHT_DIR,
        "prefixPrompt": mp.PREFIX_PROMPT,
        "pmtCacheOperation": mp.PMT_CACHE_OP,
        "pfxInitTokenLen": mp.PFX_INIT_TOKEN_LEN,
        "loraCfgPath": mp.LORA_CFG_PATH,
    }


def _common_sentence_params(max_gen_tokens):
    if mp is None:
        raise ValueError("You must call set_model_config(mp) first.")
    return {
        "callbackFreq": mp.CALLBACK_FREQ,
        "sampleFlag": mp.SAMPLE_FLAG,
        "seed": mp.SEED,
        "topK": mp.TOPK,
        "topP": mp.TOPP,
        "temperature": mp.TEMPERATURE,
        "maxGenTokens": max_gen_tokens,
        "repetitionPenalty": mp.REPETITIONPENALTY,
        "initTokenLen": mp.INIT_TOKEN_LEN,
        "isAsync": mp.IS_ASYNC,
        "stopSeq": list(mp.STOP_SEQ),
        "enableThinking": enable_thinking_flag(),
    }


def create_omc_test(test_case_name, prompt, expect="", max_gen_tokens=5000):
    sentence = {"prompt": prompt}
    sentence.update(_common_sentence_params(max_gen_tokens))
    sentence["expect"] = expect
    return {
        "testCaseName": test_case_name,
        **_common_top_level(),
        "sentences": [sentence],
    }
