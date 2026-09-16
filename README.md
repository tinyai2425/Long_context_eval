# LongBench-Pro 长上下文精度评估（GPU / OMC / API）

参照 `Ceval_ref` 的目录形态，按 `Multi_model_img_eval` 的 Qwen3.5 采样信封，对 **LongBench-Pro** 做 GPU（vLLM）、OMC、鸿蒙 API 三套同题评估。

同一 `project-N` 里三条链路的题目、prompt 文本、采样数值相同；请求信封不同。评分走 LongBench-Pro 官方指标（NDCG / Pairwise / Accuracy / F1 / SubEM / Summary），再按 primary_task、secondary_task（vertical）以及 length / language / difficulty / contextual_requirement 做 breakdown。

## 仓库结构

```
Long_context_eval/
├── README.md
├── requirements.txt
├── model_config/Qwen3.5-4B-test-config.json
├── model_config/Qwen3.5-9B-test-config.json
├── Data_set/LongBench-Pro/          ← 已下载，不进 git
├── 1_Data_gen/                      ← 长度过滤 + 写 GPU/OMC/API
├── 2_GPU_run_eval/                  ← 读 GPU jsonl，打 vLLM
├── 3_OMC_eval/                      ← 解析 OMC 日志再评分
└── 4_API_eval/
    ├── Machine_test/run_chat.sh     ← 拷到鸿蒙：只有 curl
    └── Eval_API_results.py
```

生成结果写到仓库外的 `../model-eval-storage/{model_name}/project-N/`。

## 一、环境

推荐 Python 3.10+。评测机安装：

```bash
pip install -r requirements.txt
```

GPU token 数来自 vLLM `usage`。API 有 usage 就用。OMC 优先日志里的 `inputTokenCount` / `outputTokenCount`。

鸿蒙 PC **不要**装上述 Python 包，也 **不要** python。那边只需要 `curl`。

Tokenizer 默认读 `../AIPC_LLM_eval_supplyment/Model_file/Qwen3.5-4B`（4B / 9B 共用）。也可拷到 `Model_file/Qwen3.5-4B` 后改 config。

### vLLM

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/Qwen3.5-4B \
    --served-model-name qwen35_4b \
    --host 0.0.0.0 --port 8895 --max-model-len 32768
```

内网评估前先 `unset http_proxy https_proxy all_proxy`。

## 二、生成用例（1_Data_gen）

编辑 `1_Data_gen/run_gen_longbench.sh`：

```bash
CONFIG_PATH="../model_config/Qwen3.5-4B-test-config.json"
MAX_CASE_LENGTH=27000      # chat template 后的 prompt token 上限；0 = 不过滤
MAX_CASES_PER_VERTICAL=0   # 每个 secondary_task 最多 N 条；0 = 全收
ENABLE_THINKING=false
```

`MAX_CASE_LENGTH` 对应 KV cache 预算。例如 cache=32K、`max_tokens=5000` 时配 27000：官方 `token_length` 桶大于该值的题直接丢掉，再对剩下的题用 tokenizer 数真实长度，超了也丢掉，**不做中间截断**。

```bash
cd 1_Data_gen
bash run_gen_longbench.sh
```

产出示例：

```
../model-eval-storage/Qwen3.5-4B/project-1/
  project-1-LBP-GPU-412.jsonl
  project-1-LBP-API-412.jsonl
  project-1-LBP-OMC-412.json
  eval_meta.jsonl
  sampling_config.json
  api_config.json
```

Prompt 与官方一致：`{context}\n\n\n\n{question}`。thinking 开则用 `question_thinking`，否则 `question_nonthinking`。

GPU jsonl（vLLM Chat Completions，`messages` 带 role，不是裸 prompt）：

```json
{
  "model": "Qwen3.5-4B",
  "messages": [{"role": "user", "content": "<context>\\n\\n\\n\\n<question>"}],
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": 0.7,
  "top_p": 0.8,
  "presence_penalty": 1.5,
  "seed": 99,
  "max_tokens": 5000,
  "extra_body": {
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": false}
  },
  "testCaseName": "project-1-lbp-test-T3_2_Single-Hop_Fact_QA-12",
  "expect": ["..."],
  "primary_task": "T3. Evidence-Grounded QA",
  "secondary_task": "T3.2 Single-Hop Fact QA"
}
```

API jsonl 同一套 `messages`；`enable_thinking` 在顶层（鸿蒙）。OMC json 把同一条 `messages` 用 tokenizer chat template 渲成 `sentences[0].prompt`。

## 三、GPU 评测（2_GPU_run_eval）

改 `run_eval_gpu.sh` 顶部的 jsonl 路径、vLLM IP/端口/served-model-name。长上下文默认 `MAX_WORKERS=2`，避免并发把显存打满。

```bash
cd 2_GPU_run_eval
bash run_eval_gpu.sh
```

结果落到同一 project 下的日期目录，同日重跑自动加 `-2`、`-3`：

```
.../project-1/GPU-LBP-YYYYMMDD/
  GPU-Results-1.parquet
  GPU_Summary.xlsx
  analysis_v1.png
  summary_GPU_avg.md
  category_GPU_avg.md
  breakdown_GPU_avg.md
```

`VERSION_FLAG>1` 时连跑多轮再对 sheet 求平均。

## 四、OMC 评测（3_OMC_eval）

把 `*-LBP-OMC-*.json` 放到 OMC 上跑，把日志 txt 拷回 project 目录（与 `eval_meta.jsonl` 同级），然后：

```bash
cd 3_OMC_eval
# 改 run_eval_omc.sh 里的 txt / tokenizer 路径
bash run_eval_omc.sh
```

## 五、API 评测（4_API_eval）

### 5.1 鸿蒙 PC 上跑推理

只拷这两样：

```
project-1-LBP-API-412.jsonl
4_API_eval/Machine_test/run_chat.sh
```

```sh
sh run_chat.sh /path/to/project-1-LBP-API-412.jsonl /path/to/project-1-LBP-API-412.txt
```

换地址：`API_URL=http://127.0.0.1:11434/v1/chat/completions sh run_chat.sh ...`

日志每条 3 行：请求 jsonl / 压成一行的响应 / `API_total_time: <秒>`。

### 5.2 拷回日志后评分

```bash
cd 4_API_eval
bash run_eval_api.sh
```

端侧 curl 没有 TTFT 切分，`TTFT` 为 NaN，`TPS` 用整段 `API_total_time` 近似。精度、repeat、entropy 与 GPU / OMC 同一套规则。

## 六、评分口径

与官方 [LongBench-Pro](https://github.com/caskcsg/longcontext/tree/main/LongBench-Pro) 一致：从 `[Answer]` / `[答案]` 截答案区，再按 secondary_task 选指标。

| 指标 | 任务 |
| --- | --- |
| NDCG | T1.1 / T1.2 |
| Pairwise_Accuracy | T2.1 / T2.2 / T6.3 |
| Accuracy | T3.* / T11.* |
| Summary | T4.*（默认 Rouge-L；`--embedding_model` 指向 Qwen3-Embedding 时改为 0.5 cosine + 0.5 Rouge-L） |
| F1_Score | T5.* / T6.2 / T7.* / T9.* |
| SubEM | T6.1 / T8.* / T10.* |

报表里 **SCORE / ACC** 是官方 metric 均值 ×100；**PASS** 按官方 pass@n：Summary > 0.65，其余 metric == 1.0。

- **category** sheet：11 个 primary_task
- **breakdown** sheet：25 个 secondary_task（vertical）
- 另有 token_length / language / difficulty / contextual_requirement

## 七、换模型

复制一份 `model_config`，改 `model_name`、采样、OMC 的 `MODEL_PATH` / `TOKENIZER_TYPE` 等。Qwen3.5-9B 已有模板，tokenizer 与 4B 共用；OMC 的 `MODEL_PATH` 按实际 omc 文件名改。

新模型只要仍走 Chat Completions + 同一套 `sampling` 字段即可，不必改生成/评分代码。

## 八、常见问题

**Q: 为什么 GPU 用例不是 Ceval 那种只放 user prompt 的 sentences？**  
A: Qwen3.5 走 Chat Completions。GPU jsonl 与 API 一样带 `messages: [{role: user, content: ...}]`，vLLM 自己套 chat template。OMC 则预先 `apply_chat_template`，保证三端看到同一条 user 内容。

**Q: 32K cache 为什么默认丢掉 32k 桶？**  
A: 官方 `token_length=32k` 表示样本本身约 32K tokens，再加上 chat template 和 5000 输出会超 cache。27000 上限会留下 8k / 16k 桶（再按真实 token 复核）。

**Q: 想换 9B？**  
A: `run_gen_longbench.sh` 的 `CONFIG_PATH` 改成 `Qwen3.5-9B-test-config.json`，GPU 的 served-model-name 改成对应名字。

**Q: GPU 和 API 分数差很多？**  
A: 先确认是同一次 `project-N` 里成对的 GPU/API jsonl，再核对端侧 `model` 名和 `enable_thinking`。
