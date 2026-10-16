import datetime
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.abspath("Function"))

import anlz_cont_quality
import eval_results
import extend_metrics
import extract_name
import infer_case
import parallel_fetch_gpu
import verify_ans
import write_average_sheet


USAGE = (
    "Usage: python Eval_GPU_results.py "
    "<gpu_jsonl_path> <ip> <port> <model_id> <version_flag> "
    "[--show_detail] [--max_workers N] [--embedding_model PATH] [--official_metrics]"
)


def _flag_val(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


def parse_args():
    if len(sys.argv) < 6:
        print(USAGE)
        sys.exit(1)
    jsonl_path = sys.argv[1]
    ip = sys.argv[2]
    port = int(sys.argv[3])
    model_id = sys.argv[4]
    version_flag = int(sys.argv[5])
    show_detail = "--show_detail" in sys.argv
    max_workers = int(_flag_val("--max_workers") or 2)
    embedding_model = _flag_val("--embedding_model")
    official = "--official_metrics" in sys.argv
    return (
        jsonl_path,
        ip,
        port,
        model_id,
        version_flag,
        show_detail,
        max_workers,
        embedding_model,
        official,
    )


def resolve_gpu_output(project_base, extra_runs):
    """当天的结果进同一个 GPU-LBP-日期 目录，多次启动接着往后续号。

    以前每次启动都新建 GPU-LBP-日期-2，目录里永远只有 GPU-Results-1.parquet，
    10 次运行不会被平均到一起。
    """
    date_str = datetime.date.today().strftime("%Y%m%d")
    gpu_output_dir = os.path.join(project_base, f"GPU-LBP-{date_str}")
    os.makedirs(gpu_output_dir, exist_ok=True)
    existing = []
    for name in os.listdir(gpu_output_dir):
        matched = re.fullmatch(r"GPU-Results-(\d+)\.parquet", name)
        if matched:
            existing.append(int(matched.group(1)))
    start = max(existing) + 1 if existing else 1
    version_nums = list(range(start, start + extra_runs))
    return gpu_output_dir, sorted(existing), version_nums


def rebuild_workbook(gpu_output_dir, save_summary_path, show_detail):
    paths = []
    for name in os.listdir(gpu_output_dir):
        matched = re.fullmatch(r"GPU-Results-(\d+)\.parquet", name)
        if matched:
            paths.append((int(matched.group(1)), os.path.join(gpu_output_dir, name)))
    paths.sort()
    if not paths:
        print(f"[WARN] no GPU-Results parquet in {gpu_output_dir}")
        return

    writer = pd.ExcelWriter(save_summary_path, mode="w", engine="openpyxl")
    try:
        for version_num, path in paths:
            print(f"[INFO] report version {version_num} from {os.path.basename(path)}")
            df_results = pd.read_parquet(path)
            eval_results.evaluate_reference_results(
                df_results,
                verbal=show_detail,
                save_summary_path=save_summary_path,
                save_writer=writer,
                sheet_prefix=f"v{version_num}",
            )
    finally:
        writer.close()

    for prefix in (
        "summary",
        "category",
        "breakdown",
        "token_length",
        "cat_x_len",
        "task_x_len",
    ):
        write_average_sheet.save_overall_summary(save_summary_path, prefix, mode="GPU")


if __name__ == "__main__":
    (
        jsonl_path,
        ip,
        port,
        model_id,
        version_flag,
        show_detail,
        max_workers,
        embedding_model,
        official,
    ) = parse_args()
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    verify_ans.set_overgen_penalty(not official)
    verify_ans.init_embedding(embedding_model)
    infer_case.verify_ans = verify_ans
    infer_case.extend_metrics = extend_metrics

    project_base, file_name = extract_name.parse_project_base_and_filename(jsonl_path)
    gpu_output_dir, already, version_nums = resolve_gpu_output(project_base, version_flag)
    save_summary_path = os.path.join(gpu_output_dir, "GPU_Summary.xlsx")

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] jsonl: {jsonl_path}")
    print(f"[INFO] vLLM endpoint: http://{ip}:{port}/v1, model={model_id}")
    print(f"[INFO] max_workers={max_workers}")
    print(
        "[INFO] overgen 稀释: "
        + ("off（官方口径）" if official else "on（NDCG/SubEM 按 k/n_pred 稀释）")
    )
    print(f"[INFO] Output directory: {gpu_output_dir}")
    print(
        f"[INFO] already saved versions: {already or 'none'}; "
        f"this launch adds {version_nums}"
    )

    for version_num in version_nums:
        print(f"\n[INFO] Processing version {version_num}...")
        ref_result_parquet_path = os.path.join(
            gpu_output_dir, f"GPU-Results-{version_num}.parquet"
        )
        df_results = parallel_fetch_gpu.parallel_fetch_reference_model(
            jsonl_path,
            ip,
            port,
            model_id,
            max_workers=max_workers,
            version_num=version_num,
        )
        df_results.to_parquet(ref_result_parquet_path)
        print(f"[SAVE] Results saved to {ref_result_parquet_path}")
        if show_detail:
            anlz_cont_quality.analyze_entropy_and_repeat(
                df_results,
                save_fig_path=os.path.join(
                    gpu_output_dir, f"analysis_v{version_num}.png"
                ),
            )
        print(f"[DONE] Evaluation for version {version_num}")

    rebuild_workbook(gpu_output_dir, save_summary_path, show_detail)
