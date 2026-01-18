import base64
import json
from pathlib import Path

import litellm

from medical_uncertainty.data_classes import RadiologyAssessment
from medical_uncertainty.lite_llm_caller import LitellmCaller


class GeminiInference:

    def __init__(self, model_name:str|None = None, llm_caller=None) -> None:
        self._model_name = model_name or "gemini/gemini-3-pro-preview"
        self.llm_caller= llm_caller or LitellmCaller()

    def get_model_name(self):
        return self._model_name

    def infer(self, image_path: str):
        encoded_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the patient's chest x-ray, answer the following questions:",
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Enlarged Cardiomediastinum?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Cardiomegaly?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have a Lung Lesion?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have  Lung Opacity?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Edema?"
                    },
                    #TODO: check this, consolidation
                    {
                        "type": "text",
                        "text": "Does this patient have Consolidation?"

                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Pneumonia?"

                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Atelectasis?"

                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Pneumothorax?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Pleural Effusion?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have other  Pleural conditions except effusion?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have Fracture?"
                    },
                    {
                        "type": "text",
                        "text": "Does this patient have any support devices?"
                    },


                    {
                        "type": "file",
                        "file": {
                            "file_data": "data:image/jpeg;base64,{}".format(encoded_data),
                        }
                    }

                ]
            }
        ]

        result = litellm.completion(model=self._model_name, messages=messages, response_format=RadiologyAssessment)

        return json.loads(result["choices"][0]["message"]["content"])
