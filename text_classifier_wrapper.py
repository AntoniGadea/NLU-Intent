# text_classifier_wrapper.py
import json
import os
import torch
import mlflow
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextClassifierWrapper(mlflow.pyfunc.PythonModel):
    """
    Custom MLflow PyFunc wrapper for text classification models.
    Handles tokenization and preprocessing of text input.
    """

    def load_context(self, context):
        # ... (this part remains the same)
        model_path = context.artifacts["pytorch_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with open(os.path.join(model_path, "enhanced_metadata.json"), 'r') as f:
            meta = json.load(f)
            self.id2label = {int(k): v for k, v in meta["id2label"].items()}
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    # --- Change: Added type hints to the predict method ---
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Predict intent from text input.

        Args:
            context: MLflow context
            model_input: pandas DataFrame with a 'text' column

        Returns:
            pandas DataFrame with 'intent' and 'confidence' columns
        """
        texts = model_input['text'].tolist()
        predictions = []
        confidences = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits

            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            pred_id = probabilities.argmax().item()

            intent = self.id2label[pred_id]
            confidence = probabilities[pred_id].item()

            predictions.append(intent)
            confidences.append(confidence)

        return pd.DataFrame({
            'intent': predictions,
            'confidence': confidences
        })