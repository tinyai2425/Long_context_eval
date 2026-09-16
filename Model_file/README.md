# Tokenizer（OMC 渲染 chat template / 按 token 过滤用例长度）

默认 `model_config` 指向旁边仓库里已有的 Qwen3.5-4B tokenizer：

```
../AIPC_LLM_eval_supplyment/Model_file/Qwen3.5-4B
```

Qwen3.5-4B 与 9B 共用同一套 tokenizer。若希望本目录自包含，把该目录拷到这里：

```
Long_context_eval/Model_file/Qwen3.5-4B/
```

然后把 `model_config` 里的 `TOKENIZER_CONFIG_PATH` 改成 `../Model_file/Qwen3.5-4B`。
