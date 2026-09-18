import os
import sys

sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))
sys.path.append(os.path.abspath("../3_OMC_eval/Function"))

import API_collect
import anlz_cont_quality
import eval_results
import extend_metrics
import extract_name
import file_utils
import verify_ans


USAGE = (
    "Usage: python Eval_API_results.py <test_result_txt> "
    "[--show_detail] [--embedding_model PATH]"
)


def _flag_val(flag):
    if flag not in sys.argv:
        return None
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


def parse_args():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(1)
    test_result_path = sys.argv[1]
    show_detail = "--show_detail" in sys.argv
    embedding_model = _flag_val("--embedding_model")
    return test_result_path, show_detail, embedding_model


if __name__ == "__main__":
    test_result_path, show_detail, embedding_model = parse_args()
    if not os.path.isfile(test_result_path):
        raise FileNotFoundError(test_result_path)

    verify_ans.init_embedding(embedding_model)
    project_base, file_name = extract_name.parse_project_base_and_filename(test_result_path)
    api_output_dir = extract_name.make_dated_output_dir(project_base, "API", dataset="LBP")

    API_collect.GET_answer = extend_metrics.GET_answer
    API_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    API_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    API_collect.score_case = verify_ans.score_case
    API_collect.parse_test_case_name = verify_ans.parse_test_case_name
    API_collect.attach_task_fields = verify_ans.attach_task_fields
    API_collect.iter_lines_safely = file_utils.iter_lines_safely

    print(f"[INFO] input txt    : {test_result_path}")
    print(f"[INFO] output dir   : {api_output_dir}")

    df_results = API_collect.parse_llm_api_results(test_result_path)
    if len(df_results) == 0:
        print("[ERROR] No test cases parsed; check the log format.")
        sys.exit(1)

    parquet_path = os.path.join(api_output_dir, "API-Results.parquet")
    df_results.to_parquet(parquet_path)
    print(f"[SAVE] Results saved to {parquet_path}")

    save_summary_path = os.path.join(api_output_dir, "API_Summary.xlsx")
    eval_results.evaluate_reference_results(
        df_results,
        verbal=show_detail,
        save_summary_path=save_summary_path,
        sheet_prefix="api",
    )
    if show_detail:
        anlz_cont_quality.analyze_entropy_and_repeat(
            df_results, save_fig_path=os.path.join(api_output_dir, "analysis.png")
        )
