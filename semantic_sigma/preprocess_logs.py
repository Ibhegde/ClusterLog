import os
from nltk.util import ngrams
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt
import re


@DeprecationWarning
def preprocess_doc_old(log_text):
    # Tokenize the text to get words with '.' and '-' but without '_'
    tk = RegexpTokenizer(r"([^\W_]+\.?-?[^\W_]+)")
    tokens = tk.tokenize(log_text)

    # Apply stop words
    # includes:
    # repeated characters
    # starting with numbers
    log_stopwords_reg = re.compile(r"\w*((\w)\2{2,})|(\d+.*)")
    tokens = [
        word
        for word in tokens
        if len(word) <= 16 and log_stopwords_reg.match(word) is None
    ]

    return tokens


def preprocess_doc(log_text: str):
    # Tokenize the text to get words with '.','-', and ':'
    tk = RegexpTokenizer(r"([\w\-\.:]+)")
    tokens = tk.tokenize(log_text)

    # Apply stop words
    # Full word match:
    # repeated characters
    # starting with non word characters
    # Ending with non-word characters
    # Single character
    log_stopwords_full = re.compile(
        r".*((\w)\2{2,}).*|([\W\d_]+.*)|(.*[\-\.:_])|(\b\w\b)"
    )

    # Partial match:
    # IP address/domain names (fe80::7b9c, intel.com), labeles (mmu0:), and file names (filename.log)
    log_stopwords_reg = re.compile(r".*[\.:]+.*")
    tokens = [
        word
        for word in tokens
        if len(word) <= 32
        and log_stopwords_reg.match(word) is None
        and log_stopwords_full.fullmatch(word) is None
    ]

    return tokens


@DeprecationWarning
def tokenise_doc_old(
    log_text: str,
    is_truncate: bool = False,
    stopwords_file: str = "../analysis/intel_log_stopwords_tfidf.csv",
) -> list[str]:
    # Tokenize the text to get words with '.' and '-' but without '_'
    tk = RegexpTokenizer(r"([^\W_]+\.?-?[^\W_]+)")
    tokens = tk.tokenize(log_text)

    # Remove stop words
    # includes:
    # repeated characters
    # starting with numbers
    log_stopwords_reg = re.compile(r"\w*((\w)\2{2,})|(\d+.*)")

    # Stopwords evalaueated before
    stopwords = pd.read_csv(stopwords_file, header=None)
    stopwords = stopwords[0].tolist()

    tokens = [
        word
        for word in tokens
        if len(word) <= 16
        and log_stopwords_reg.match(word) is None
        and word not in stopwords
    ]

    if is_truncate and len(tokens) > 512:
        tokens = tokens[-512:]

    return tokens if len(tokens) > 10 else []


def tokenise_doc(
    log_text: str, stopwords_file: str = "analysis/intel_log_stopwords.csv"
) -> list[str]:
    # Tokenize the text to get words with '.','-', and ':'
    # New addition capture newline
    tk = RegexpTokenizer(r"([\w\-\.:]+)")
    tokens = tk.tokenize(log_text)

    # Apply stop words
    # Full word match:
    # repeated characters
    # starting with non word characters
    # Ending with non-word characters
    # Single character
    log_stopwords_full = re.compile(
        r".*((\w)\2{2,}).*|([\W\d_]+.*)|(.*[\-\.:_])|(\b\w\b)"
    )

    # Partial match:
    # IP address/domain names (fe80::7b9c, intel.com), labeles (mmu0:), and file names (filename.log)
    log_stopwords_reg = re.compile(r".*[\.:]+.*")

    # Stopwords evalaueated before
    stopwords = pd.read_csv(stopwords_file, header=None)
    stopwords = stopwords[0].tolist()

    tokens = [
        word
        for word in tokens
        if 3 <= len(word) <= 32
        and log_stopwords_reg.match(word) is None
        and log_stopwords_full.fullmatch(word) is None
        and word not in stopwords
    ]

    return tokens


def file_ext_analysis(logs_dir: str, analysis_dir: str):

    file_ext_freq = {}

    for filename in os.listdir(logs_dir):
        fname_split = filename.split(".")
        f_ext = "NA"
        if len(fname_split) > 1:
            f_ext = fname_split[-1]

        if f_ext in file_ext_freq:
            file_ext_freq[f_ext] += 1
        else:
            file_ext_freq[f_ext] = 1

    file_ext_df = pd.DataFrame.from_dict(file_ext_freq, orient="index")
    file_ext_df.to_csv(os.path.join(analysis_dir, "ext_freq.csv"))


def load_data(logs_dir: str, max_rec: int = None):
    file_ext_excpetion_regx = [
        r".*\.jpeg",
        r".*\..ini.*",
        r".*\.conf.*",
        r".*\..*_conf_.*",
        r".*\.json",
        r".*\.pwd",
        r".*\.alias",
        r".*_Pinmaps_.*",
        r".*_meminfo",
        r".*\.deny",
        r".*\.dat",
        r".*\.itf",
        r".*\.stat",
    ]
    combined_execption_regx = "(" + ")|(".join(file_ext_excpetion_regx) + ")"

    docs = pd.DataFrame(columns=["ci_number", "log_filename", "content"])

    for filename in os.listdir(logs_dir):
        if max_rec == 0:
            break
        if re.match(combined_execption_regx, filename):
            # Exclude these files
            continue

        ci_number = filename[:11]

        # Open the text file
        # print('Opening.. ' + filename)
        with open(
            os.path.join(logs_dir, filename), "r", encoding="utf-8", errors="ignore"
        ) as lf:
            log_text = lf.read()
            log_entry = pd.DataFrame(
                [
                    {
                        "ci_number": ci_number,
                        "log_filename": filename,
                        "content": log_text,
                    }
                ]
            )
            docs = pd.concat([docs, log_entry], ignore_index=True)
            if max_rec is not None:
                max_rec = max_rec - 1
    return docs


def main():
    # TODO: Check if this function is needed in the future

    LOGS_DIR = "C:/Users/bhegde/codes/SemanticSigma/data/Sigma_project/train"

    pd_docs = load_data(logs_dir=LOGS_DIR, max_rec=100)
    tfidf = TfidfVectorizer(
        analyzer="word",
        tokenizer=preprocess_doc,
        stop_words=None,
        max_df=0.3,
        min_df=0.05,
    )
    tfidf.fit_transform(pd_docs["content"])

    pd_docs = pd.DataFrame(tfidf.get_feature_names_out())
    pd_docs.to_csv("analysis/tfidf_vocab_main.csv", index=False)


if __name__ == "__main__":
    main()
