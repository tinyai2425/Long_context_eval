"""LongBench-Pro 报表：overall + primary_task + secondary_task，以及 length/language/difficulty。"""

import pandas as pd

import verify_ans

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


def _label(value, default="unknown"):
    if verify_ans._blank(value):
        return default
    return str(value).strip()


def _gold_lines(df):
    if "expect" not in df.columns:
        return None
    return df["expect"].map(lambda e: len(verify_ans.parse_expect(e)))


def _dump_rate(df):
    """答案行数超过 gold 两倍的样本占比。

    量化后掉的主要是这个：同一题 GPU 输出 10 行、OMC 把 1..N 全打出来。
    SCORE 只差一两分看不出来，这一列能看出来。
    """
    gold = _gold_lines(df)
    if gold is None or "n_pred_lines" not in df.columns:
        return 0.0
    valid = gold > 0
    if not valid.any():
        return 0.0
    over = df.loc[valid, "n_pred_lines"] > 2 * gold[valid]
    return round(over.mean(), 4)


def _row(df, project_name, flavor, vertical=""):
    metric_mean = df["metric"].mean() if "metric" in df.columns else 0.0
    gold = _gold_lines(df)
    return {
        "Project": project_name,
        "Flavor": flavor,
        "Vertical": vertical,
        "Test No.": len(df),
        "prompt len": round(df["prompt_token_len"].mean(), 2),
        "response len": round(df["response_token_len"].mean(), 2),
        "pred lines": round(df["n_pred_lines"].mean(), 2)
        if "n_pred_lines" in df.columns
        else 0.0,
        "gold lines": round(gold.mean(), 2) if gold is not None else 0.0,
        "dump>2x": _dump_rate(df),
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


def _has_values(df, col):
    if col not in df.columns:
        return False
    return df[col].map(lambda v: _label(v, "") != "").any()


def _group_rows(df, project_name, key, order=None):
    work = df.copy()
    work[key] = work[key].map(_label)
    rows = []
    grouped = work.groupby(key, dropna=False)
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


def _prepare_report_df(df):
    prepared = df.copy()
    records = []
    for row in prepared.to_dict(orient="records"):
        records.append(verify_ans.attach_task_fields(row))
    return pd.DataFrame(records)


def evaluate_reference_results(
    result_get,
    verbal=False,
    save_summary_path=None,
    save_writer=None,
    sheet_prefix="",
):
    df_results = _prepare_report_df(result_get)
    project_name = "LongBench-Pro"
    if "project_name" in df_results.columns and len(df_results):
        project_name = _label(df_results["project_name"].iloc[0], project_name)

    df_summary = pd.DataFrame([_row(df_results, project_name, "overall")])

    extra_sheets = {}
    if _has_values(df_results, "token_length"):
        extra_sheets["token_length"] = _group_rows(
            df_results, project_name, "token_length", verify_ans.LENGTH_BUCKET_ORDER
        )
    df_category = None
    df_breakdown = None
    if _has_values(df_results, "primary_task"):
        df_category = _group_rows(df_results, project_name, "primary_task", PRIMARY_ORDER)
    vertical_key = None
    if _has_values(df_results, "secondary_task"):
        vertical_key = "secondary_task"
    elif _has_values(df_results, "vertical"):
        vertical_key = "vertical"
    if vertical_key:
        df_breakdown = _group_rows(df_results, project_name, vertical_key)
    for dim in ("language", "difficulty", "contextual_requirement"):
        if _has_values(df_results, dim):
            extra_sheets[dim] = _group_rows(df_results, project_name, dim)

    print(f"\n[Summary] {sheet_prefix}")
    print(df_summary.to_string(index=False))

    if "token_length" in extra_sheets:
        print(f"\n[token_length / 8k-16k] {sheet_prefix}")
        print(extra_sheets["token_length"].to_string(index=False))
    else:
        print(
            f"\n[WARN] no token_length breakdown for {sheet_prefix}: "
            "8k/16k labels missing"
        )

    if df_category is not None:
        print(f"\n[Primary task] {sheet_prefix}")
        print(df_category.to_string(index=False))
    if df_breakdown is not None:
        print(f"\n[Breakdown / {vertical_key}] {sheet_prefix}")
        print(df_breakdown.to_string(index=False))
    else:
        print(
            f"\n[WARN] no vertical breakdown for {sheet_prefix}: "
            "secondary_task/vertical missing"
        )
    if verbal:
        for dim, table in extra_sheets.items():
            if dim == "token_length":
                continue
            print(f"\n[{dim}] {sheet_prefix}")
            print(table.to_string(index=False))

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
        if df_category is not None:
            df_category.to_excel(save_writer, sheet_name=category_sheet, index=False)
            print(f"[Excel] Category written to sheet: {category_sheet}")
        if df_breakdown is not None:
            df_breakdown.to_excel(save_writer, sheet_name=breakdown_sheet, index=False)
            print(f"[Excel] Breakdown written to sheet: {breakdown_sheet}")
        for dim, table in extra_sheets.items():
            sheet = (dim[: 31 - len(suffix)] + suffix)[:31]
            table.to_excel(save_writer, sheet_name=sheet, index=False)
            print(f"[Excel] {dim} written to sheet: {sheet}")

    if internal_writer:
        internal_writer.close()
    return df_results
