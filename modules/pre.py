import os, torch, csv
from PIL import Image
from torchvision import transforms
from torchtext import data

def csv_to_dict(file_path, label_num, is_test=False):
    title_dict = {}
    label_dict = {}
    if is_test:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = row.pop(reader.fieldnames[1])
                title_dict[key] = row.pop(reader.fieldnames[2])
        return title_dict
    else:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = row.pop(reader.fieldnames[1])
                title_dict[key] = row.pop(reader.fieldnames[2])
                label_dict[key] = label_num[row.pop(reader.fieldnames[3])]
        return title_dict, label_dict

def total_count(directory):
    total_file_num = 0
    for root, dirs, files in os.walk(directory):
        total_file_num = total_file_num + files.__len__()
    return total_file_num

def preprocess(directory, image_size, tokenizer, vocab, max_seq_length, titles, labels, batch_size):
    image_tensors = []
    text_tensors = []
    answer_tensors = []

    iter_sizes = create_list_with_sum(batch_size, batch_size//2)

    preprocess_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    for (root, dirs, files), iter in zip(os.walk(directory), iter_sizes):
        if files.__len__() < iter: continue
        samples = random.sample(files, iter)
        for file in samples:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as image:
                        # 이미지 전처리 및 텐서로 변환
                        tensor = preprocess_image(image.convert("RGB"))
                        image_tensors.append(tensor)
                    tensor = preprocess_text(titles[file], tokenizer, vocab, max_seq_length)
                    text_tensors.append(tensor)
                    tensor = labels[file]
                    answer_tensors.append(tensor)

                except IOError:
                    print(f"이미지를 열 수 없습니다: {image_path}")

    # 이미지 텐서를 하나의 텐서로 결합
    images = torch.stack(image_tensors).cuda()
    texts = torch.stack(text_tensors).cuda()
    answers = torch.tensor(answer_tensors).cuda()

    return images, texts, answers

def preprocess_for_test(directory, image_size, tokenizer, vocab, max_seq_length, titles, batch_size, st_idx):
    image_tensors = []
    text_tensors = []

    preprocess_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    for id in range(st_idx, st_idx + batch_size):
        if id > 29435:
            break
        file = f"{id}.jpg"
        image_path = os.path.join(directory, file)
        try:
            with Image.open(image_path) as image:
                # 이미지 전처리 및 텐서로 변환
                tensor = preprocess_image(image.convert("RGB"))
                image_tensors.append(tensor)
            tensor = preprocess_text(titles[file], tokenizer, vocab, max_seq_length)
            text_tensors.append(tensor)
        except IOError:
            print(f"이미지를 열 수 없습니다: {image_path}")
                

    # 이미지 텐서를 하나의 텐서로 결합
    images = torch.stack(image_tensors).cuda()
    texts = torch.stack(text_tensors).cuda()

    return images, texts

def preprocess_text(sentence, tokenizer, vocab, max_length):
    tokenized = tokenizer(sentence)  # 문장을 토큰화
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokenized]  # 토큰을 정수로 인코딩

    # 시퀀스 길이 맞추기
    padded = encoded[:max_length] + [0] * max(0, max_length - len(encoded))

    # 텐서로 변환
    tensor = torch.tensor(padded)

    return tensor

def make_vocabulary(titles, tokenizer):
    vocab = {}
    counter = 2
    for title in titles:
        for token in tokenizer(title):
            if token not in vocab:
                vocab[token] = counter
                counter = counter + 1
    vocab["<UNK>"] = 1
    return vocab

def make_label_number(label_list):
    label_num = {}
    counter = 0
    for label in label_list:
        label_num[label] = counter
        counter = counter + 1
    return label_num

def one_hot_encode(targets, num_classes):
    # targets: 정답 텐서
    # num_classes: 클래스의 개수
    
    batch_size = targets.size(0)
    one_hot_targets = torch.zeros(batch_size, num_classes)
    one_hot_targets.scatter_(1, targets.view(-1, 1), 1)
    
    return one_hot_targets

import os
import random
import shutil

import os


def create_folders(label_list, parent_folder):
    parent_folder = "C:\\Codes\\newjeansNet\\data\\jbnu-swuniv-ai\\val"  # 폴더를 생성할 부모 폴더 경로를 지정해주세요

    for label in label_list:
        folder_name = label.replace(", ", "_")  # 폴더 이름에 쉼표와 공백을 언더스코어로 대체합니다
        folder_path = os.path.join(parent_folder, label)
        os.makedirs(folder_path, exist_ok=True)
        print(f"Created folder: {folder_name}")

def move_files_in_folders(label_list, percentage, from_path, to_path):
    for folder_name in label_list:
        folder_path = os.path.join(
            from_path, folder_name
        )  # 부모 폴더 경로와 각 폴더 이름을 조합하여 경로 생성
        file_list = os.listdir(folder_path)

        num_files = int(len(file_list) * percentage)

        files_to_move = random.sample(file_list, num_files)
        for file_name in files_to_move:
            source_path = os.path.join(folder_path, file_name)
            destination_path = os.path.join(
                to_path, folder_name, file_name
            )  # 이동할 폴더 경로 생성
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name}")

        print(f"Total files moved from {folder_name}: {num_files}")

import random

def create_list_with_sum(target_sum, max_value):
    # Generate 24 random numbers within the specified range
    numbers = [random.randint(1, max_value) for _ in range(24)]

    # Calculate the current sum
    current_sum = sum(numbers)

    # Adjust the numbers to achieve the target sum
    while current_sum != target_sum:
        if current_sum > target_sum:
            # Reduce a random number
            index = random.randint(0, 23)
            diff = min(numbers[index], current_sum - target_sum)
            numbers[index] -= diff
            current_sum -= diff
        else:
            # Increase a random number
            index = random.randint(0, 23)
            diff = min(max_value - numbers[index], target_sum - current_sum)
            numbers[index] += diff
            current_sum += diff

    return numbers