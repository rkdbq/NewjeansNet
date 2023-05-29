import torch
import torch.nn as nn
import torch.optim as optim

# 멀티 모달 분류 모델 정의
class MultiModalModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, hidden_dim, num_classes):
        super(MultiModalModel, self).__init__()

        # 텍스트 모듈 정의
        self.text_embedding = nn.Embedding(text_input_dim, 100)
        self.text_fc = nn.Linear(100, hidden_dim)

        # 이미지 모듈 정의
        self.image_conv = nn.Conv2d(image_input_dim, 16, kernel_size=3, stride=1, padding=1)
        self.image_relu = nn.ReLU()
        self.image_fc = nn.Linear(16 * 128 * 128, hidden_dim)

        # 멀티 모달 분류기 정의
        self.fc = nn.Linear(hidden_dim * 33, num_classes)

    def forward(self, text_input, image_input):
        # 텍스트 입력 처리
        text_embedded = self.text_embedding(text_input)
        text_output = self.text_fc(text_embedded).squeeze(0)

        # 이미지 입력 처리
        image_output = self.image_conv(image_input)
        image_output = self.image_relu(image_output)
        image_output = image_output.view(image_output.size(0), -1)
        image_output = self.image_fc(image_output).unsqueeze(1)

        # 텍스트와 이미지 특성을 연결하여 멀티 모달 분류 수행

        combined_features = torch.cat((text_output, image_output), dim=1)
        combined_features = combined_features.view(combined_features.size(0), -1)
        output = self.fc(combined_features)

        return output