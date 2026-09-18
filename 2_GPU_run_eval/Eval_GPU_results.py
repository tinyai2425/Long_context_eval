import os
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
    "[--show_detail] [--max_workers N] [--embedding_model PATH]"
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
    return jsonl_path, ip, port, model_id, version_flag, show_detail, max_workers, embedding_model


def get_version_nums(version_flag):
    if version_flag < 1:
        raise ValueError("version_flag must be >= 1")
    return list(range(1, version_flag + 1))


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
    ) = parse_args()
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    verify_ans.init_embedding(embedding_model)
    infer_case.verify_ans = verify_ans
    infer_case.extend_metrics = extend_metrics

    project_base, file_name = extract_name.parse_project_base_and_filename(jsonl_path)
    version_nums = get_version_nums(version_flag)
    gpu_output_dir = extract_name.make_dated_output_dir(project_base, "GPU", dataset="LBP")
    save_summary_path = os.path.join(gpu_output_dir, "GPU_Summary.xlsx")

    print("Notice: Proxy settings should be disabled before proceeding.")
    print(f"[INFO] jsonl: {jsonl_path}")
    print(f"[INFO] vLLM endpoint: http://{ip}:{port}/v1, model={model_id}")
    print(f"[INFO] max_workers={max_workers}")
    print(f"[INFO] Output directory: {gpu_output_dir}")

    writer = pd.ExcelWriter(save_summary_path)
    try:
        for version_num in version_nums:
            print(f"\n[INFO] Processing version {version_num}...")
            ref_result_parquet_path = os.path.join(
                gpu_output_dir, f"GPU-Results-{version_num}.parquet"
            )
            df_results = parallel_fetch_gpu.parallel_fetch_reference_model(
                jsonl_path, ip, port, model_id, max_workers=max_workers
            )
            df_results.to_parquet(ref_result_parquet_path)
            print(f"[SAVE] Results saved to {ref_result_parquet_path}")

            eval_results.evaluate_reference_results(
                df_results,
                verbal=show_detail,
                save_summary_path=save_summary_path,
                save_writer=writer,
                sheet_prefix=f"v{version_num}",
            )
            if show_detail:
                anlz_cont_quality.analyze_entropy_and_repeat(
                    df_results,
                    save_fig_path=os.path.join(
                        gpu_output_dir, f"analysis_v{version_num}.png"
                    ),
                )
            print(f"[DONE] Evaluation for version {version_num}")
    finally:
        if writer is not None:
            writer.close()

    write_average_sheet.save_overall_summary(save_summary_path, "summary", mode="GPU")
    write_average_sheet.save_overall_summary(save_summary_path, "category", mode="GPU")
    write_average_sheet.save_overall_summary(save_summary_path, "breakdown", mode="GPU")
    write_average_sheet.save_overall_summary(save_summary_path, "token_length", mode="GPU")
