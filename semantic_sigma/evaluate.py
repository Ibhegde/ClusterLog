import os
import re
import shutil
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd

import concurrent.futures as cfu

known_error_regex = {
    "rcs0": r"rcs0",
    "GPU Hang": r"GPU Hang",
    "FIFO underrun": r"pipe A FIFO underrun",
    "MCE L1 Cache": r"Machine Check Exception.*1101.*5",
    "Flip Done": r"flip_done",
    "Buffer IO": r"Buffer I/O error",
    "Kernel panic not syncing": r"Kernel panic - not syncing",
    "Segmentation Fault Kernel RIP": r"Segmentation fault",
    "general protection fault": r"general protection fault:.*PREEMPT SMP NOPTI",
    "DMAR: DRHD: handling fault": r"DMAR: .*DMA Read.* Request device.*fault addr",
    "IPU-isys, FIFO Overflow": r"csi2-0 error: FIFO Overflow",
    "IPU-isys, csi2-0 fatal error": r"csi2-0 received fatal error",
    "IPU-isys, csi2-0 not initialized": r"IPU4 CSI-2.*was not initialized!",
}


def extract_errors(log_path: str, keywords: list[str], result_file: str = None):
    """
    Extract log lines from `log_path` consisting of provided `keywords`.
    Extracted log lines will be written to `result_file`, if provided,.

    Args:
        log_path (str): input log file path.
        keywords (list[str]): list of keywords.
        result_file (str, optional): File path to write results. Defaults to None.

    Returns:
        str: log lines containing keywords
    """
    # Uses grep to seach patterns
    command = (
        ["C:/Program Files (x86)/GnuWin32/bin/grep.exe", "-ainE"]
        + ["|".join([kw.strip() for kw in keywords])]
        + [log_path]
    )

    # Use Silver searcher 'ag'
    # Example: ag -ai 'entity|e3900|n3350|series|mmc1|processing|dai|codec|signal|tdf8532-hifi'  -G 'CI1920-8545_LHCRB logs_putty-unit2-usb-pcie-room.log' 'data/Sigma_project/train/'
    # source_dir = os.path.dirname(log_path)
    # source_fname = os.path.basename(log_path)
    # command = ['ag', '-ai'] + ['|'.join([kw.strip() for kw in keywords]) ] + ['-G', source_fname, source_dir]

    # print("Running command: [" + " ".join(command) + "]")
    p_result = subprocess.run(command, capture_output=True)
    result = p_result.stdout.decode(encoding="utf-8", errors="ignore")

    # Remove empty lines in the result
    result = "\n".join([eln for eln in result.split("\n") if len(eln.rstrip()) > 0])

    if result_file is not None:
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        with open(result_file, "w", errors="ignore", newline="\n") as rf:
            rf.write(result)
    return result


def find_errors(log_dir, debug_count=None):
    log_error_list = pd.DataFrame(columns=["log_filename", "errors"])

    for idx, filename in enumerate(os.listdir(log_dir)):
        if debug_count is not None:
            debug_count = debug_count - 1
            if debug_count < 0:
                break

        err_list = []
        for err_key in known_error_regex.keys():
            log_path = os.path.join(log_dir, filename)
            err_result = extract_errors(log_path, [known_error_regex[err_key]])
            if len(err_result) > 0:
                err_list.append(err_key)
        log_error_list.loc[idx] = [filename, err_list]

    return log_error_list


def find_errors_parallel(log_dir: str, debug_count: int = None):
    log_err_list = {}
    log_proc = {}

    with cfu.ThreadPoolExecutor() as executor:
        for err_key, error_reg in known_error_regex.items():
            for idx, filename in enumerate(os.listdir(log_dir)):
                if debug_count is not None and debug_count <= idx:
                    break
                log_path = os.path.join(log_dir, filename)
                ext_proc = executor.submit(extract_errors, log_path, [error_reg])
                log_proc[ext_proc] = (filename, err_key)

        for ext_proc in cfu.as_completed(log_proc):
            err_result = ext_proc.result()
            filename, err_key = log_proc[ext_proc]
            if filename not in log_err_list:
                log_err_list[filename] = []
            if len(err_result) > 0:
                log_err_list[filename].append(err_key)

    df_error_res = pd.DataFrame(
        log_err_list.items(), columns=["log_filename", "errors"]
    )

    df_error_res["errors"] = df_error_res["errors"].map(lambda el: ", ".join(el))

    return df_error_res


def get_logline_cont(log_dir):
    log_error_lines = pd.DataFrame(columns=["log_filename", "error_lines"])

    for idx, filename in enumerate(os.listdir(log_dir)):
        log_path = os.path.join(log_dir, filename)
        with open(log_path, "r", errors="ignore") as lf:
            log_error_lines.loc[idx] = [
                filename,
                len([ln for ln in lf.readlines() if len(ln.strip()) > 0]),
            ]

    return log_error_lines


def extract_logs_from_keywords(keyword_ref, log_dir, extract_dir, debug_count=None):
    keyword_result = pd.read_csv(keyword_ref)
    keyword_result.reset_index()
    keyword_result = keyword_result.fillna("")

    if debug_count is not None:
        keyword_result = keyword_result.loc[: debug_count - 1, :]

    log_full_path_lst = keyword_result["log_filename"].apply(
        lambda lf: os.path.join(log_dir, lf)
    )
    extract_keywords_lst = keyword_result["key_words"].apply(
        lambda kw_lst: kw_lst.split(",")
    )
    result_full_path_lst = keyword_result["log_filename"].apply(
        lambda lf: os.path.join(extract_dir, lf)
    )

    with cfu.ThreadPoolExecutor() as executor:
        executor.map(
            extract_errors,
            log_full_path_lst,
            extract_keywords_lst,
            result_full_path_lst,
        )


def extract_lines(source_path: int, result_path: int, line_count: int):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(source_path, "r", errors="ignore") as lf, open(
        result_path, "w", errors="ignore"
    ) as rf:
        rf.writelines(lf.readlines()[-line_count:])


def extract_loglines(
    filename_ref, log_dir, extract_dir, line_count: int = 100, debug_count=None
):
    keyword_result = pd.read_csv(filename_ref)
    keyword_result.reset_index()
    keyword_result = keyword_result.fillna("")

    if debug_count is not None:
        keyword_result = keyword_result.loc[: debug_count - 1, :]

    log_full_path_lst = keyword_result["log_filename"].apply(
        lambda lf: os.path.join(log_dir, lf)
    )
    result_full_path_lst = keyword_result["log_filename"].apply(
        lambda lf: os.path.join(extract_dir, lf)
    )
    line_count_list = [line_count] * keyword_result.shape[0]

    # Extract log line
    with cfu.ThreadPoolExecutor() as executor:
        executor.map(
            extract_lines, log_full_path_lst, result_full_path_lst, line_count_list
        )


def evaluate(
    force_extract: bool = False,
    original_log_dir: str = None,
    filtered_log_dir: str = None,
    keyword_result: str = None,
    orig_log_result: str = None,
    debug_count: int = None,
    line_count: int = None,
):
    """Evaluate log extracts based on:
        1. Number of known errors captured.
        2. Percentage of log lines reduced in the log extracts.

    Args:
        original_log_dir (str, optional): Full path to original log files. Defaults to None.
        filtered_log_dir (str, optional): Full path to log extracts. Defaults to None.
        keyword_result (str, optional): CSV consisting of keywords for each log document. Defaults to None.
    """
    if len(sys.argv) in [3, 4]:
        original_log_dir = "data/Sigma_project/" + sys.argv[1] + "/"
        # Extract filename from the keyword results path
        keyword_result = sys.argv[2]

        kwr_fname = os.path.splitext(os.path.basename(keyword_result))[0]
        filtered_log_dir = "data/" + kwr_fname + "_extract/"
        orig_log_result = "data/Sigma_project/" + sys.argv[1] + "_labels.csv"

    if line_count is not None:
        force_extract = True

    if force_extract and os.path.exists(filtered_log_dir):
        shutil.rmtree(filtered_log_dir)

    eval_start = time.time()
    if not os.path.exists(filtered_log_dir):
        if line_count is not None:
            extract_loglines(
                keyword_result,
                original_log_dir,
                filtered_log_dir,
                debug_count=debug_count,
                line_count=line_count,
            )
        else:
            extract_logs_from_keywords(
                keyword_result,
                original_log_dir,
                filtered_log_dir,
                debug_count=debug_count,
            )

    # Evaluate number of matches with known regex identifiers
    orig_err_list = pd.read_csv(orig_log_result)
    orig_err_list = orig_err_list.fillna("")

    filt_err_list = find_errors_parallel(
        log_dir=filtered_log_dir, debug_count=debug_count
    )

    print("\nCompleted evaluation in %.2f sec\n" % (time.time() - eval_start))

    error_score = orig_err_list.merge(filt_err_list, on="log_filename")

    es_x = error_score["errors_x"].map(
        lambda el: set([e.strip() for e in el.split(",")])
    )
    es_y = error_score["errors_y"].map(
        lambda el: set([e.strip() for e in el.split(",")])
    )
    error_score["error_diff"] = es_x - es_y
    error_score["error_diff"] = error_score["error_diff"].map(lambda el: ", ".join(el))

    filt_lines = get_logline_cont(log_dir=filtered_log_dir)
    error_score = error_score.merge(filt_lines, on="log_filename")

    el_x = error_score.error_lines_x
    el_y = error_score.error_lines_y
    error_score["line_reduction"] = el_y.div(el_x).mul(100).round(2)

    kwr_fname = os.path.splitext(os.path.basename(keyword_result))[0]
    kwr_full_path = os.path.join(
        os.path.dirname(keyword_result), kwr_fname + "_errordiff.csv"
    )
    error_score.to_csv(kwr_full_path, index=False)

    # Plot the result to get a clear idea
    print("Percentage of errors number mismatch:")
    err_diff_len = error_score["error_diff"].apply(
        lambda el: len([e for e in el.split(",") if len(e.strip()) > 0])
    )
    res_err_diff = err_diff_len.value_counts(normalize=True).mul(100).round(2)
    print(res_err_diff)

    res_reduced_score = error_score["line_reduction"].mean()
    print("Average error lines extracted: %.2f %%" % (res_reduced_score))

    return res_err_diff, res_reduced_score


if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate(force_extract=True)
        # evaluate(force_extract=True, debug_count=6)
    elif len(sys.argv) == 4:
        evaluate(line_count=int(sys.argv[3]))
        # evaluate(debug_count=6, line_count=int(sys.argv[3]))
    else:
        print("Usage: evaluate.py train/test keyword_file [extract line count]")
        exit(-1)
