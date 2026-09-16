"""LongBench-Pro 报表：overall + primary_task + secondary_task，以及 length/language/difficulty。"""

import pandas as pd

PRIMARY_ORDER = [
    "T1. Retrieval & Ranking",
    "T2. Sequencing & Structure Reconstruction",
    "T3. Evidence-Grounded QA",
    "T4. Summarization & Synthesis",
    "T5. Attribution & Citation Alignment",
    "T6. Aggregation & Clustering",
    "T7. Consistency & Compliance Checking",
    "T8. Structured & Numeric Reasoning",
    "T9. Version & Code Diff Analysis",
    "T10. Rule Induction & In-Context Learning",
    "T11. Dialogue Memory & Long-Horizon Tracking",
]


def _safe_tps(df):
    decode_sum = df["decode_time"].dropna().sum()
    if decode_sum <= 0:
        total = df["total_time"].sum()
        ftt = df["first_token_time"].dropna().sum()
        decode_sum = max(1e-5, total - ftt)
    return df["response_token_len"].sum() / decode_sum


def _row(df, project_name, flavor, vertical=""):
    metric_mean = df["metric"].mean() if "metric" in df.columns else 0.0
    return {
        "Project": project_name,
        "Flavor": flavor,
        "Vertical": vertical,
        "Test No.": len(df),
        "prompt len": round(df["prompt_token_len"].mean(), 2),
        "response len": round(df["response_token_len"].mean(), 2),
        "get_ans": round(df["get_ans"].mean(), 2) if "get_ans" in df.columns else 0.0,
        "rp>0.1": round((df["repeat"] > 0.1).mean(), 4),
        "rp@99%": round(df["repeat"].quantile(0.99), 5),
        "ent<3.5": round((df["entropy"] < 3.5).mean(), 4),
        "ent@1%": round(df["entropy"].quantile(0.01), 2),
        "TTFT": round(df["first_token_time"].mean(), 2),
        "TPS": round(_safe_tps(df), 2),
        "SCORE": round(metric_mean * 100, 2),
        "PASS": round(df["correct"].mean() * 100, 2),
        "ACC": round(metric_mean * 100, 2),
    }


def _group_rows(df, project_name, key, order=None):
    rows = []
    grouped = df.groupby(key, dropna=False)
    names = list(grouped.groups.keys())
    if order:
        ordered = [n for n in order if n in grouped.groups]
        rest = [n for n in names if n not in ordered]
        names = ordered + rest
    for name in names:
        sub = grouped.get_group(name)
        row = _row(df=sub, project_name=project_name, flavor=str(name), vertical=str(name))
        row["样本数"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix="",
):
    df_results = result_get.copy()
    project_name = "LongBench-Pro"
    if "project_name" in df_results.columns and len(df_results):
        project_name = df_results["project_name"].iloc[0] or project_name

    df_summary = pd.DataFrame([_row(df_results, project_name, "overall")])
    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    df_category = None
    df_breakdown = None
    extra_sheets = {}
    if verbal:
        if "primary_task" in df_results.columns:
            print(f"\n[Primary task] {sheet_prefix}")
            df_category = _group_rows(df_results, project_name, "primary_task", PRIMARY_ORDER)
            print(df_category.to_string(index=False))

        vertical_key = "secondary_task" if "secondary_task" in df_results.columns else "vertical"
        print(f"\n[Breakdown / {vertical_key}] {sheet_prefix}")
        df_breakdown = _group_rows(df_results, project_name, vertical_key)
        print(df_breakdown.to_string(index=False))

        for dim in ("token_length", "language", "difficulty", "contextual_requirement"):
            if dim in df_results.columns:
                extra_sheets[dim] = _group_rows(df_results, project_name, dim)
                print(f"\n[{dim}] {sheet_prefix}")
                print(extra_sheets[dim].to_string(index=False))

    internal_writer = None
    if save_summary_path and save_writer is None:
        internal_writer = pd.ExcelWriter(save_summary_path)
        save_writer = internal_writer

    if save_writer:
        suffix = f"_{sheet_prefix}" if sheet_prefix else ""
        summary_sheet = ("summary" + suffix)[:31]
        category_sheet = ("category" + suffix)[:31]
        breakdown_sheet = ("breakdown" + suffix)[:31]

        df_summary.to_excel(save_writer, sheet_name=summary_sheet, index=False)
        print(f"[Excel] Summary written to sheet: {summary_sheet}")
        if verbal and df_category is not None:
            df_category.to_excel(save_writer, sheet_name=category_sheet, index=False)
            print(f"[Excel] Category written to sheet: {category_sheet}")
        if verbal and df_breakdown is not None:
            df_breakdown.to_excel(save_writer, sheet_name=breakdown_sheet, index=False)
            print(f"[Excel] Breakdown written to sheet: {breakdown_sheet}")
        for dim, table in extra_sheets.items():
            sheet = (dim[: 31 - len(suffix)] + suffix)[:31]
            table.to_excel(save_writer, sheet_name=sheet, index=False)
            print(f"[Excel] {dim} written to sheet: {sheet}")

    if internal_writer:
        internal_writer.close()
    return df_results
