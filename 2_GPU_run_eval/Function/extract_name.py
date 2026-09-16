import datetime
import os
import re


def parse_project_base_and_filename(input_path):
    abs_path = os.path.abspath(input_path)
    return os.path.dirname(abs_path), os.path.basename(abs_path)


def make_dated_output_dir(project_base, mode, dataset="LBP", date_str=None):
    if date_str is None:
        date_str = datetime.date.today().strftime("%Y%m%d")
    base = f"{mode}-{dataset}-{date_str}"
    candidate = base
    n = 1
    while os.path.exists(os.path.join(project_base, candidate)):
        n += 1
        candidate = f"{base}-{n}"
    full_path = os.path.join(project_base, candidate)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def extract_count_from_filename(filename):
    match = re.search(r"-([0-9]+)\.[a-zA-Z0-9]+$", filename)
    if not match:
        return None
    return int(match.group(1))
