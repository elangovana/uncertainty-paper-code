# uncertainty-paper-code


```bash
export PYTHONPATH=./src
export "Set model api key"
python src/medical_uncertainty/workflow.py --model_name gpt-5.1 --images_dir "<chexpertimagesidr>"
```


# Compute existing scores
```bash
python src/medical_uncertainty/compute_scores.py --data_dir redacted_results/results_sonnet/multinli_matched --ground_truth_file datasets/mnli/multinli_1.0_dev_matched.jsonl.small.jsonl --output_file_path=temp/mnli

```