import os
import re

import pandas as pd
from tabulate import tabulate


def _version_sheets(sheet_names, prefix):
    matched = []
    for name in sheet_names:
        found = re.fullmatch(rf"{re.escape(prefix)}_v(\d+)", name)
        if found:
            matched.append((int(found.group(1)), name))
    matched.sort()
    return [name for _, name in matched]


def save_overall_summary(save_summary_path, prefix, mode="GPU"):
    xls = pd.ExcelFile(save_summary_path)
    target_sheets = _version_sheets(xls.sheet_names, prefix)
    if not target_sheets:
        print(f"No {prefix}_vN sheet found in {save_summary_path}")
        return

    print(f"Aggregating {len(target_sheets)} {prefix} sheets: {target_sheets}")
    df_list = [pd.read_excel(xls, sheet_name=s) for s in target_sheets]
    combined = pd.concat(df_list, ignore_index=True)

    group_keys = [c for c in ["Project", "Flavor", "Vertical"] if c in combined.columns]
    # Excel 把空单元格读成 NaN。summary 的 Vertical 以前是空的，
    # groupby 默认丢掉 NaN 键，整张总表就变成只有表头。
    for key in group_keys:
        combined[key] = combined[key].where(combined[key].notna(), "")
    numeric_cols = (
        combined.select_dtypes(include="number").columns.difference(group_keys).tolist()
    )
    if not numeric_cols:
        print(f"No numeric columns to average for {prefix}")
        return
    df_avg = (
        combined.groupby(group_keys, dropna=False)[numeric_cols].mean().reset_index()
    )
    df_avg["runs"] = len(target_sheets)

    df_first = pd.read_excel(xls, sheet_name=target_sheets[0])
    for key in group_keys:
        if key in df_first.columns:
            df_first[key] = df_first[key].where(df_first[key].notna(), "")
    for col in combined.columns:
        if col not in numeric_cols + group_keys and col in df_first.columns:
            values = df_first[[*group_keys, col]]
            df_avg = pd.merge(df_avg, values, on=group_keys, how="left")

    ordered_cols = [c for c in df_first.columns if c in df_avg.columns]
    if "runs" not in ordered_cols:
        ordered_cols.append("runs")
    df_avg = df_avg[ordered_cols]
    xls.close()

    avg_sheet = f"{prefix}_avg"[:31]
    with pd.ExcelWriter(
        save_summary_path,
        mode="a",
        engine="openpyxl",
        if_sheet_exists="replace",
    ) as writer:
        df_avg.to_excel(writer, sheet_name=avg_sheet, index=False)

    md_path = os.path.join(
        os.path.dirname(save_summary_path), f"{prefix}_{mode}_avg.md"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(tabulate(df_avg, headers="keys", tablefmt="github"))
    print(f"Markdown summary saved: {md_path}")
