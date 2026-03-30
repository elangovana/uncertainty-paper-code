import json
import os
from collections import Counter

import fire
import pandas as pd
from sklearn.metrics import accuracy_score
import  structlog

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

def compute_scores_by_pd_mnli(data_dir: str, ground_truth_file: str, output_file_path: str) -> dict:
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
    df_final = ground_truth_df.merge(model_df, left_on=["pairID", "promptID"], right_on=["pairID", "promptID"], how="left")
    df_final["pd"] = df_final["annotator_labels"].apply(lambda x: round(Counter(x).most_common()[0][1] / len(x), 3))
    df_final["model_prediction"] = df_final["geval"].apply(
        lambda x: label_mappings.get(_parse_output(x), UNKNOWN_PREDICTION))

    scores_by_pd = {}
    for pd_unique in list(df_final["pd"].unique()):
        df_by_pd = df_final.query("pd=={}".format(pd_unique))
        valid_values_in_gt = set(df_by_pd["gold_label"].unique()) - set( label_mappings.values())
        if len(valid_values_in_gt) > 0:
            _logger.info("No valid values in gt, hence skipping", invalid_vals = valid_values_in_gt, pd_unique = pd_unique)
            continue
        print(df_by_pd["gold_label"].unique())
        print(df_by_pd["model_prediction"].unique())
        accuracy = accuracy_score(df_by_pd["gold_label"], df_by_pd["model_prediction"])
        scores_by_pd[pd_unique] = {"score": accuracy, "count": len(df_by_pd)}

    datadir_prefix = "_".join(data_dir.split(os.path.sep)[-4:])
    scores = {
        datadir_prefix: {
            "_".join(ground_truth_file.split(os.path.sep)[-4:]): scores_by_pd
        }
    }

    input_file_prefix = os.path.splitext(datadir_prefix)[0]

    gt_prefix = os.path.splitext(os.path.basename(ground_truth_file))[0]

    file_name = os.path.join(output_file_path, f"{input_file_prefix}_{gt_prefix}_scores_by_pd.json")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    print(f"Writing output to {file_name}")

    with open(file_name, "w") as f:
        f.write(json.dumps(scores, indent=4))
    return scores


if __name__ == "__main__":
    fire.Fire(compute_scores_by_pd_mnli)
