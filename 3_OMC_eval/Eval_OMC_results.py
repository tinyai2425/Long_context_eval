import os
import sys

sys.path.append(os.path.abspath("Function"))
sys.path.append(os.path.abspath("../2_GPU_run_eval/Function"))

import OMC_collect
import anlz_cont_quality
import eval_results
import extend_metrics
import extract_name
import file_utils
import verify_ans


USAGE = (
    "Usage: python Eval_OMC_results.py <tokenizer_path> <test_result_path> "
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
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)
    tokenizer_path = sys.argv[1]
    test_result_path = sys.argv[2]
    show_detail = "--show_detail" in sys.argv
    embedding_model = _flag_val("--embedding_model")
    return tokenizer_path, test_result_path, show_detail, embedding_model


if __name__ == "__main__":
    tokenizer_path, test_result_path, show_detail, embedding_model = parse_args()
    if not os.path.isfile(test_result_path):
        raise FileNotFoundError(test_result_path)

    verify_ans.init_embedding(embedding_model)
    project_base, file_name = extract_name.parse_project_base_and_filename(test_result_path)
    omc_output_dir = extract_name.make_dated_output_dir(project_base, "OMC", dataset="LBP")

    OMC_collect.init_tokenizer(tokenizer_path)
    OMC_collect.GET_answer = extend_metrics.GET_answer
    OMC_collect.calculate_repetition_rate = extend_metrics.calculate_repetition_rate
    OMC_collect.calculate_token_entropy = extend_metrics.calculate_token_entropy
    OMC_collect.score_case = verify_ans.score_case
    OMC_collect.TEST_CASE_NAME_PATTERN = verify_ans.TEST_CASE_NAME_PATTERN
    OMC_collect.parse_test_case_name = verify_ans.parse_test_case_name
    OMC_collect.attach_task_fields = verify_ans.attach_task_fields
    OMC_collect.iter_lines_safely = file_utils.iter_lines_safely

    print(f"[INFO] input txt    : {test_result_path}")
    print(f"[INFO] tokenizer    : {tokenizer_path}")
    print(f"[INFO] output dir   : {omc_output_dir}")

    df_results = OMC_collect.parse_llm_test_results(test_result_path)
    if len(df_results) == 0:
        print("[ERROR] No test cases parsed; check the log format.")
        sys.exit(1)

    parquet_path = os.path.join(omc_output_dir, "OMC-Results.parquet")
    df_results.to_parquet(parquet_path)
    print(f"[SAVE] Results saved to {parquet_path}")

    save_summary_path = os.path.join(omc_output_dir, "OMC_Summary.xlsx")
    eval_results.evaluate_reference_results(
        df_results,
        verbal=show_detail,
        save_summary_path=save_summary_path,
        sheet_prefix="omc",
    )
    if show_detail:
        anlz_cont_quality.analyze_entropy_and_repeat(
            df_results, save_fig_path=os.path.join(omc_output_dir, "analysis.png")
        )
