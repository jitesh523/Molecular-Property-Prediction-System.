import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class SMILESTransformer(nn.Module):
    """
    A wrapper around HuggingFace Transformers (like ChemBERTa)
    for predicting molecular properties from SMILES text.
    """
    def __init__(
        self,
        model_name: str,
        num_tasks: int,
        dropout: float = 0.1,
    ):
        super(SMILESTransformer, self).__init__()
        self.model_name = model_name
        self.num_tasks = num_tasks
        
        # Load the base model.
        # num_labels maps directly to the number of tasks in the regression/classification head.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_tasks,
            classifier_dropout=dropout
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass using tokenized text input.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # SequenceClassification models return logits directly
        return outputs.logits

    def get_device(self):
        """Helper to get the model's device."""
        return next(self.model.parameters()).device
