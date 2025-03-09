import torch
import torch.nn as nn

from transformers import BertModel


class SentimentClassifier(nn.Module):
    """
    Sentiment Classification Model using BERT as a feature extractor.
    """

    def __init__(
        self,
        n_classes: int,
        model_name: str = "bert-base-uncased",
        dropout_prob: float = 0.3,
    ):
        """
        Initializes the Sentiment Classifier.

        Args:
            n_classes (int): Number of output classes (e.g., 5 for sentiment classification).
            model_name (str): Pretrained BERT model name.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tokenized input IDs.
            attention_mask (torch.Tensor): Attention mask for padding.

        Returns:
            torch.Tensor: Logits (raw scores before softmax).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Extract [CLS] token output
        dropped_output = self.dropout(pooled_output)
        return self.fc(dropped_output)  # Final classification layer
