import torch
import torch.nn as nn
import torch.nn.functional as F

# 멀티 모달 분류 모델 정의
class MultiModalModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, hidden_dim, num_classes):
        super(MultiModalModel, self).__init__()

        # 텍스트 모듈 정의
        self.text_embedding = nn.Embedding(text_input_dim, 100).cuda()
        self.text_fc = nn.Linear(100, hidden_dim).cuda()
        self.text_bn = nn.BatchNorm1d(hidden_dim*2).cuda()
        self.text_relu = nn.ReLU().cuda()
        self.text_dropout = nn.Dropout(0.5).cuda()

        # 이미지 모듈 정의
        self.image_conv1 = nn.Conv2d(image_input_dim, 32, kernel_size=3, stride=1, padding=1).cuda()
        self.image_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1).cuda()
        self.image_maxpool = nn.MaxPool2d(kernel_size=2, stride=2).cuda()
        self.image_relu = nn.ReLU().cuda()
        self.image_fc = nn.Linear(16 * 16 * 16, hidden_dim).cuda()
        self.image_bn = nn.BatchNorm1d(1).cuda()
        self.image_dropout = nn.Dropout(0.5).cuda()

        # 멀티 모달 분류기 정의
        self.fc = nn.Linear(hidden_dim * 33, num_classes).cuda()

    def forward(self, text_input, image_input):
        # 텍스트 입력 처리
        text_input = text_input.cuda()
        text_embedded = self.text_embedding(text_input)
        text_output = self.text_fc(text_embedded).squeeze(0)
        text_output = self.text_bn(text_output)
        text_output = self.text_relu(text_output)
        text_output = self.text_dropout(text_output)

        # 이미지 입력 처리
        image_input = image_input.cuda()
        image_output = self.image_conv1(image_input)
        image_output = self.image_relu(image_output)
        image_output = self.image_conv2(image_output)
        image_output = self.image_relu(image_output)
        image_output = self.image_maxpool(image_output)
        image_output = image_output.view(image_output.size(0), -1)
        image_output = self.image_fc(image_output).unsqueeze(1)
        image_output = self.image_bn(image_output)
        image_output = self.image_relu(image_output)
        image_output = self.image_dropout(image_output)

        # 텍스트와 이미지 특성을 연결하여 멀티 모달 분류 수행

        combined_features = torch.cat((text_output, image_output), dim=1)
        combined_features = combined_features.view(combined_features.size(0), -1)
        output = self.fc(combined_features)

        return output