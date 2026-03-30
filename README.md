# uncertainty-paper-code


```bash
export PYTHONPATH=./src
export "Set model api key"
python src/medical_uncertainty/workflow.py --model_name gpt-5.1 --images_dir "<chexpertimagesidr>"
```


# Compute existing scores
```bash
python src/medical_uncertainty/compute_scores.py --data_base_dir redacted_results --ground_truth_files_dir . --output_file_path=temp/mnli

```