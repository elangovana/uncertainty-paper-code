import json
import os
from collections import Counter
from pathlib import Path

import fire
import pandas as pd
from sklearn.metrics import accuracy_score
import structlog
import matplotlib.pyplot as plt
from statistics import median
from medical_uncertainty.combined_delta_plots import create_multimodel_pd_boxplot_by_df_acc

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


def _process_nli(df_final, label_mappings, output_file_name):
    df_final["pd"] = df_final["annotator_labels"].apply(lambda x: round(Counter(x).most_common()[0][1] / len(x), 2))
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
    scores_by_pd = _process_nli(df_final, label_mappings, output_file_name)

    return scores_by_pd


def compute_scores_by_pd_snli(data_dir: str, ground_truth_file: str, output_file_name: str) -> dict:
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
    df_final = ground_truth_df.merge(model_df, left_on=["pairID", "captionID"], right_on=["pairID", "captionID"],
                                     how="left")
    scores_by_pd = _process_nli(df_final, label_mappings, output_file_name)

    return scores_by_pd


def compute_scores_by_pd_summeval(data_file: str, ground_truth_file: str, output_file_name: str, criteria) -> dict:
    with open(data_file, "r") as f:
        model_df = pd.DataFrame(json.load(f))
    ground_truth_df = pd.read_json(ground_truth_file, lines=True)
    df_final = ground_truth_df.merge(model_df, left_on=["id", "model_id"], right_on=["doc_id", "system_id"],
                                     how="inner")
    df_final["gold_label"] = df_final[f"expert_annotations_{criteria}"].apply(lambda x: Counter(x).most_common()[0][0])

    df_final["pd"] = df_final[f"expert_annotations_{criteria}"].apply(
        lambda x: round(Counter(x).most_common()[0][1] / len(x), 2))
    df_final["model_prediction"] = df_final[f"all_responses"].apply(lambda x: int(Counter([xi.strip()[0] for xi in x]).most_common()[0][0]))

    scores_by_pd = {}
    for pd_unique in list(df_final["pd"].unique()):
        df_by_pd = df_final.query("pd=={}".format(pd_unique))
        valid_values_in_gt = set(df_by_pd["gold_label"].unique()) - {1, 2, 3, 4, 5}
        invalid_values_in_model =    set(df_final["model_prediction"].unique()) -  {1, 2, 3, 4, 5}

        if len(valid_values_in_gt) > 0:
            _logger.info("No valid values in gt, hence skipping", invalid_vals=valid_values_in_gt, pd_unique=pd_unique)
            continue
        if len(invalid_values_in_model) > 0:
            _logger.warning("In valid values in model, hence skipping", unique_model_val = invalid_values_in_model, pd_unique=pd_unique)
            continue
        print(f"================",criteria,  pd_unique, "============")
        print("Model", df_by_pd["model_prediction"].value_counts())
        print("gold_label", df_by_pd["gold_label"].value_counts())

        accuracy = accuracy_score(df_by_pd["gold_label"], df_by_pd["model_prediction"])

        scores_by_pd[pd_unique] = {
            "count": len(df_by_pd),
            "pd": pd_unique,
            "criteria": criteria,
            "score": accuracy
        }

    print(f"Writing output to {output_file_name}")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    with open(output_file_name, "w") as f:
        json.dump(scores_by_pd, f)
    return scores_by_pd


def compute_scores_by_pd_all_models_nli(data_base_dir: str, ground_truth_files_dir: str, output_file_path: str) -> None:
    mnli_matched_gt_file = os.path.join(ground_truth_files_dir, "datasets", "mnli",
                                        "multinli_1.0_dev_matched.jsonl.small.jsonl")
    mnli_mismatched_gt_file = os.path.join(ground_truth_files_dir, "datasets", "mnli",
                                           "multinli_1.0_dev_mismatched.jsonl.small.jsonl")
    snli_gt_file = os.path.join(ground_truth_files_dir, "datasets", "mnli", "snli_1.0_dev.jsonl.small.jsonl")

    mnli_datatuples = [
        (os.path.join(data_base_dir, "results_llama31_8b", "mnli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_mistral", "mnli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_mistral_large", "mnli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_sonnet", "multinli_matched"), mnli_matched_gt_file),
        (os.path.join(data_base_dir, "results_llama31_8b", "mnli_mismatched"), mnli_mismatched_gt_file),
        (os.path.join(data_base_dir, "results_mistral", "mnli_mismatched"), mnli_mismatched_gt_file),
        (os.path.join(data_base_dir, "results_mistral_large", "mnli_mismatched"), mnli_mismatched_gt_file),
        (os.path.join(data_base_dir, "results_sonnet", "multinli_mismatched"), mnli_mismatched_gt_file)

    ]

    snli_datatuples = [
        (os.path.join(data_base_dir, "results_llama31_8b", "snli"), snli_gt_file),
        (os.path.join(data_base_dir, "results_mistral", "snli"), snli_gt_file),
        (os.path.join(data_base_dir, "results_mistral_large", "snli"), snli_gt_file),
        (os.path.join(data_base_dir, "results_sonnet", "snli"), snli_gt_file),

    ]

    gt_data_processor_map = {
        mnli_mismatched_gt_file: compute_scores_by_pd_mnli,
        mnli_matched_gt_file: compute_scores_by_pd_mnli,
        snli_gt_file: compute_scores_by_pd_snli,
    }
    nli_datatuples = mnli_datatuples + snli_datatuples

    plot_data = []

    all_scores = {}
    for t in nli_datatuples:
        data_dir = t[0]
        ground_truth_file = t[1]
        datadir_model_prefix = data_dir.split(os.path.sep)[-2]
        compute_scores_by_pd_func = gt_data_processor_map[ground_truth_file]

        gt_prefix = os.path.splitext(os.path.basename(ground_truth_file))[0]
        input_file_prefix = os.path.splitext(datadir_model_prefix)[0]
        output_file_name = os.path.join(output_file_path, f"{input_file_prefix}_{gt_prefix}_scores_by_pd.json")

        scores_by_pd = compute_scores_by_pd_func(data_dir, ground_truth_file, output_file_name)
        plot_data.extend([{"pd": float(k), "score": v["score"]} for k, v in scores_by_pd.items()])

        if datadir_model_prefix not in all_scores:
            all_scores[datadir_model_prefix] = {}

        all_scores[datadir_model_prefix][ground_truth_file.split(os.path.sep)[-1]] = scores_by_pd
        with open(os.path.join(output_file_path, "all_scores_nli.json"), "w") as f:
            f.write(json.dumps(all_scores, indent=4))

    fig, _ = create_multimodel_pd_boxplot_by_df_acc(pd.DataFrame(plot_data), "score", "accuracy", pd_column_name="pd",
                                                    y_min_max=(0, 1.4), y_label="Models accuracy distribution (NLI)")
    fig.savefig(Path(output_file_path) / "multimodel_mnli_pd_boxplot_acc.png", dpi=400, bbox_inches='tight')
    plt.close(fig)


def compute_pd_summeval(data_base_dir: str, ground_truth_files_dir: str, output_file_path: str):
    summeval_gt_file = os.path.join(ground_truth_files_dir, "datasets", "summeval",
                                    "summeval_model_annotations.aligned.jsonl.small.jsonl")
    datatuples = [
        (os.path.join(data_base_dir, "geval", "gpt4_coh_detailed.json"), summeval_gt_file, "coherence"),
        (os.path.join(data_base_dir, "geval", "gpt4_con_detailed.json"), summeval_gt_file, "consistency"),
        (os.path.join(data_base_dir, "geval", "gpt4_flu_detailed.json"), summeval_gt_file, "fluency"),
        (os.path.join(data_base_dir, "geval", "gpt4_rel_detailed.json"), summeval_gt_file, "relevance")
    ]
    all_scores = {}
    plot_data = []
    for t in datatuples:
        data_dir = t[0]
        ground_truth_file = t[1]
        criteria = t[2]
        datadir_model_prefix = data_dir.split(os.path.sep)[-2]

        gt_prefix = os.path.splitext(os.path.basename(ground_truth_file))[0] + f"_{criteria}"
        input_file_prefix = os.path.splitext(datadir_model_prefix)[0]
        output_file_name = os.path.join(output_file_path, f"{input_file_prefix}_{gt_prefix}_scores_by_pd.json")

        scores_by_pd = compute_scores_by_pd_summeval(data_dir, ground_truth_file, output_file_name, criteria)
        plot_data.extend([{"pd": float(k), "delta":float(k)- v["score"]} for k, v in scores_by_pd.items()])

        if datadir_model_prefix not in all_scores:
            all_scores[datadir_model_prefix] = {}

        all_scores[datadir_model_prefix][ground_truth_file.split(os.path.sep)[-1] + f"_{criteria}"] = scores_by_pd

    with open(os.path.join(output_file_path, "all_scores_summeval.json"), "w") as f:
        f.write(json.dumps(all_scores, indent=4))

    fig,_ = create_multimodel_pd_boxplot_by_df_acc(pd.DataFrame(plot_data), "delta", "$\\Delta_{{EM}}$ ", pd_column_name="pd", y_min_max=(-0.2, 1.5),y_label="G-Eval $\\Delta_{{EM}}$ accuracy distribution ", pd_levels_color_map =  {0.33: '#d62728' , 0.67:'#ff7f0e',  1.0: '#2ca02c'}
)
    fig.savefig(Path(output_file_path) / "multimodel_geval_pd_boxplot_delta_acc.png", dpi=400, bbox_inches='tight')
    plt.close(fig)


def compute_all_datasets(data_base_dir: str, ground_truth_files_dir: str, output_file_path: str):
    # compute_scores_by_pd_all_models_nli(data_base_dir, ground_truth_files_dir, output_file_path)
    compute_pd_summeval(data_base_dir, ground_truth_files_dir, output_file_path)


if __name__ == "__main__":
    fire.Fire({"geval": compute_pd_summeval,
               "nli": compute_scores_by_pd_all_models_nli
               })
