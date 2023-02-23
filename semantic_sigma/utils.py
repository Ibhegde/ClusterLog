import os
import pandas as pd

from .evaluate import find_errors_parallel, get_logline_cont


def combine_data(log_dir):
    df_err_label = find_errors_parallel(log_dir=log_dir)
    df_err_label["ci_number"] = df_err_label["log_filename"].apply(
        lambda filename: filename[:11]
    )

    error_lines = get_logline_cont(log_dir=log_dir)
    df_err_label = df_err_label.merge(error_lines, on="log_filename")

    # Re-arrage columns
    df_err_label = df_err_label[["ci_number", "log_filename", "errors", "error_lines"]]
    df_err_label.to_csv(
        os.path.join(
            os.path.dirname(log_dir), os.path.basename(log_dir) + "_labels.csv"
        ),
        index=False,
    )


def main():
    METADATA_DIR = "C:/Users/bhegde/codes/SemanticSigma/data/Sigma_project/"

    train_log_dir = os.path.join(METADATA_DIR, "train")
    combine_data(train_log_dir)

    test_log_dir = os.path.join(METADATA_DIR, "test")
    combine_data(test_log_dir)


if __name__ == "__main__":
    main()
