import json
import os
import re

import fire
import structlog

from medical_uncertainty.gemini_inference import GeminiInference
from medical_uncertainty.openai_inference import OpenAIInference


class Workflow:

    def __init__(self, model_inference):
        self.model_inference = model_inference

    @property
    def _logger(self):
        return structlog.get_logger(__name__)

    def run_workflow(self, images_dir, output_dir, checkpoint_dir):
        results = self.load_checkpoint(checkpoint_dir)
        for p in os.listdir(images_dir):
            patient = p
            image = os.path.join(images_dir, p, "study1", "view1_frontal.jpg")
            self._logger.info("Running patient", patient=patient)

            if p in set(map(lambda x: x["patient"], results)):
                self._logger.info("Found patient in checkpoint. hence skipping", patient=patient )
                continue

            model_response = self.model_inference.infer(image)
            results.append({
                "patient": patient,
                "image_path": os.path.relpath(image, start=images_dir),
                "model_response": model_response,
                "model_name": self.model_inference.get_model_name(),

            })
            self._logger.info("Finished patient", patient=patient)
            self.dump_checkpoint(results, checkpoint_dir)

    def dump_checkpoint(self, data, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write(json.dumps(data))

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
        data = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                data = json.load(f)
        return data


def run_inference(
        model_name: str,
        images_dir: str,
):
    model_name_mapping = {
        "gpt-4o": lambda: OpenAIInference(model_name),
        "gpt-5.1": lambda: OpenAIInference(model_name),
        "gemini/gemini-3-pro-preview": lambda: GeminiInference(model_name),

    }
    model_clean_name =  re.sub(r"\\/\s+", "_", model_name)

    inferencer = model_name_mapping[model_name]()
    data_dir = images_dir
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "..", f"data_dir/checkpoints/{model_clean_name}")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", f"data_dir/output/{model_clean_name}")
    Workflow(inferencer).run_workflow(data_dir, output_dir, checkpoint_dir)



if __name__ == '__main__':
  fire.Fire(run_inference)
