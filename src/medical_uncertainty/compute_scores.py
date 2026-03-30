import json
import os
from collections import Counter

import fire
import pandas as pd
from sklearn.metrics import accuracy_score
import structlog

_logger = structlog.get_logger()
UNKNOWN_PREDICTION = "-"


def _parse_output(x: str):
    try:
        x_value = eval(x)
        if x_value:
            return Counter(x_value).most_common()[0][0]
        else:
            return UNKNOWN_PREDICTION
    except Exception:
        _logger.error(x)
        return UNKNOWN_PREDICTION


def compute_scores_by_pd_mnli(data_dir: str, ground_truth_file: str, output_file_name: str) -> dict:
    label_mappings = {
        "[[n]]": "neutral",
        "[[c]]": "contradiction",
        "[[e]]": "entailment",
    }
    all_dfs = []
    for file_name in filter(lambda x: x.endswith(".csv"), os.listdir(data_dir)):
        df = pd.read_csv(os.path.join(data_dir, file_name))
        all_dfs.append(df)
    model_df = pd.concat(all_dfs)
    ground_truth_df = pd.read_json(ground_truth_file, lines=True)
    ground_truth_df = ground_truth_df[ground_truth_df["gold_label"].isin(label_mappings.values())].copy()
    df_final = ground_truth_df.merge(model_df, left_on=["pairID", "promptID"], right_on=["pairID", "promptID"],
                                     how="left")
    df_final["pd"] = df_final["annotator_labels"].apply(lambda x: round(Counter(x).most_common()[0][1] / len(x), 3))
    df_final["model_prediction"] = df_final["geval"].apply(
        lambda x: label_mappings.get(_parse_output(x), UNKNOWN_PREDICTION))

    scores_by_pd = {}
    for pd_unique in list(df_final["pd"].unique()):
        df_by_pd = df_final.query("pd=={}".format(pd_unique))
        valid_values_in_gt = set(df_by_pd["gold_label"].unique()) - set(label_mappings.values())

        if len(valid_values_in_gt) > 0:
            _logger.info("No valid values in gt, hence skipping", invalid_vals=valid_values_in_gt, pd_unique=pd_unique)
            continue

        accuracy = accuracy_score(df_by_pd["gold_label"], df_by_pd["model_prediction"])
        scores_by_pd[pd_unique] = {"score": accuracy, "count": len(df_by_pd)}

    print(f"Writing output to {output_file_name}")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    with open(output_file_name, "w") as f:
        json.dump(scores_by_pd, f)

    return scores_by_pd


def compute_scores_by_pd_all_models(data_base_dir: str, ground_truth_files_dir: str, output_file_path: str) -> dict:
    mnli_matched_gt_file = os.path.join(ground_truth_files_dir, "datasets", "mnli",
                                        "multinli_1.0_dev_matched.jsonl.small.jsonl")
    mnli_mismatched_gt_file = os.path.join(ground_truth_files_dir, "datasets", "mnli",
                                           "multinli_1.0_dev_mismatched.jsonl.small.jsonl")

    datatuples = [
        (os.path.join(data_base_dir, "results_llama31_8b", "mnli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_mistral", "mnli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_mistral_large", "mnli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_sonnet", "multinli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_llama31_8b", "mnli_mismatched"), mnli_mismatched_gt_file),
        (os.path.join(data_base_dir, "results_mistral", "mnli_mismatched"), mnli_mismatched_gt_file),
        (os.path.join(data_base_dir, "results_mistral_large", "mnli_mismatched"), mnli_mismatched_gt_file),
        (os.path.join(data_base_dir, "results_sonnet", "multinli_mismatched"), mnli_mismatched_gt_file)

    ]
    all_scores = {}
    for t in datatuples:
        data_dir = t[0]
        ground_truth_file = t[1]
        datadir_model_prefix = data_dir.split(os.path.sep)[-2]

        gt_prefix = os.path.splitext(os.path.basename(ground_truth_file))[0]
        input_file_prefix = os.path.splitext(datadir_model_prefix)[0]
        output_file_name = os.path.join(output_file_path, f"{input_file_prefix}_{gt_prefix}_scores_by_pd.json")

        scores_by_pd = compute_scores_by_pd_mnli(data_dir, ground_truth_file, output_file_name)

        if datadir_model_prefix not in all_scores:
            all_scores[datadir_model_prefix] = {}

        all_scores[datadir_model_prefix][ground_truth_file.split(os.path.sep)[-1]] = scores_by_pd
        with open(os.path.join(output_file_path, "all_scores.json"), "w") as f:
            f.write(json.dumps(all_scores, indent=4))


if __name__ == "__main__":
    fire.Fire(compute_scores_by_pd_all_models)
