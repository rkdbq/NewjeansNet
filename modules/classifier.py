import torch
import torch.nn as nn
import torch.optim as optim

# 분류를 위한 모델 정의
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.text_fc = nn.Linear(input_size, num_classes)
        self.image_fc = nn.Linear(input_size, num_classes)

    def forward(self, text, image):
        out_text = self.text_fc(text)
        out_image = self.image_fc(image)
        return out_text + out_image

# # 데이터 준비
# input_size = 1000
# num_classes = 5

# # 모델 초기화
# model = Classifier(input_size, num_classes)

# # 로그 소프트맥스와 로그우도 손실 함수 정의
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# # 입력 데이터
# input_data = torch.randn(1, input_size)

# # 로짓 예측
# logits = model(input_data)

# # 클래스 예측
# _, predicted_classes = torch.max(logits, 1)

# print("Logits:", logits)
# print("Predicted classes:", predicted_classes)
