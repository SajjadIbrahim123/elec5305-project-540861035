import torch
import torch.nn as nn


class EmotionClassifier(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# Quick test
if __name__ == "__main__":

    model = EmotionClassifier()

    # Pretend we have one 768-dimensional speech embedding
    test_embedding = torch.randn(1, 768)

    output = model(test_embedding)

    print(model)
    print("\nInput shape:", test_embedding.shape)
    print("Output shape:", output.shape)
