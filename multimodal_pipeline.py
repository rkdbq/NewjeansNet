import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from modules.multimodal import *
from modules.pre import *
from modules.post import *
from modules.classifier import *
from torchtext import data

label_list = ["Arts, Photography", "Biographies, Memoirs", "Calendars", "Childrens Books", "Computers, Technology", "Cookbooks, Food, Wine", "Crafts, Hobbies, Home", 
              "Education, Teaching", "Engineering, Transportation", "Health, Fitness, Dieting", "Humor, Entertainment", "Law", "Literature, Fiction", "Medical Books", 
              "Mystery, Thriller, Suspense", "Parenting, Relationships", "Reference", "Religion, Spirituality", "Science Fiction, Fantasy", "Science, Math", 
              "Self Help", "Sports, Outdoors", "Test Preparation", "Travel"]

#pathes
base_path ="C:\\Codes\\newjeansNet\\"
train_data_csv_path = base_path + "data\\jbnu-swuniv-ai\\train_data.csv"
train_img_path = base_path + "data\\jbnu-swuniv-ai\\train\\"
val_img_path = base_path + "data\\jbnu-swuniv-ai\\val\\"
result_csv_path = base_path + "results.csv"

# create validation folders [PREPROCESS]
# create_folders(label_list, val_img_path)
# move_files_in_folders(label_list, 0.2, train_img_path, val_img_path)

# hyperparameters
max_seq_length = 32
resize_size = (16, 16) # image size
batch_size = 128
learning_rate = 0.001

#################

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# 데이터 준비
category_to_num = {}
num_to_category = {}
counter = 0

for label in label_list:
    category_to_num[label] = counter
    num_to_category[counter] = label
    counter = counter + 1

titles, labels = csv_to_dict(train_data_csv_path, category_to_num)
tokenizer = data.get_tokenizer("basic_english")  # 공백을 기준으로 텍스트를 토큰화
vocab = make_vocabulary(titles.values(), tokenizer)

total = total_count(train_img_path)

# 모델 초기화
text_input_dim = vocab.__len__()
image_input_dim = 3
hidden_dim = 16
num_classes = category_to_num.__len__()

model = MultiModalModel(text_input_dim, image_input_dim, hidden_dim, num_classes).cuda()

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"batch size: {batch_size}")
print(f"total data set size: {total}")
print("Learning start")

# 모델 학습
num_epochs = 1
for epoch in range(num_epochs):
    for iter in range(total//batch_size):
        image_input, text_input, labels_input = preprocess(train_img_path, resize_size, tokenizer, vocab, max_seq_length, titles, labels, batch_size)
        image_input = (image_input - image_input.mean()) / image_input.std()

        # Forward 연산
        output = model(text_input.unsqueeze(0), image_input).cuda()
        
        # Loss 계산
        criterion.cuda()
        loss = criterion(output, labels_input)
        
        # Backward 연산 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 학습 과정 출력
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 학습된 모델로 예측 수행
val_image, val_text, val_labels = preprocess(val_img_path, resize_size, tokenizer, vocab, max_seq_length, titles, labels, batch_size)
val_image = (val_image - val_image.mean()) / val_image.std()

predicted = model(val_text.unsqueeze(0), val_image)
_, predicted_labels = torch.max(predicted, 1)

accuracy = np.mean(np.equal(predicted_labels.cpu(), val_labels.cpu()).numpy())
accuracy_percentage = accuracy * 100

predicted_categories = map_numbers_to_text(predicted_labels, num_to_category)
numpy_array_to_csv(predicted_categories, result_csv_path)

print(f"Accuracy: {accuracy_percentage:.2f}%")