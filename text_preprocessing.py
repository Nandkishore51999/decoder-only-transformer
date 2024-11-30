import os
import traceback
import re
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_text_files(folder_path):
    combined_text = ""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    combined_text += f.read()
            except:
                traceback.print_exc()
    return combined_text


def text_cleaning(input_text):
    pattern = r"[¥é–—’…₩₹、《》いうきくっとま三傻坞大宝莱闹]"
    input_text = re.sub(pattern, "", input_text)

    delimiters = " -().,$[]"
    pattern = f"[{re.escape(delimiters)}]"

    all_text_tokens = re.split(pattern, input_text)
    all_text_tokens = [ele.strip() for ele in all_text_tokens]
    return all_text_tokens


def tokens_to_tensors(input_text_list):
    global encode, decode
    vocab = sorted(list(set(input_text_list)))
    vocab_size = len(vocab)
    print("Vocab Size: ", vocab_size)

    tokens_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_tokens = {idx: word for idx, word in enumerate(vocab)}

    try:
        encode = lambda s: [tokens_to_idx[c] for c in s]
        decode = lambda i: " ".join([idx_to_tokens[idx] for idx in i])
    except:
        traceback.print_exc()

    all_text_tensors = [tokens_to_idx[token] for token in input_text_list]
    return all_text_tensors, vocab_size, tokens_to_idx, idx_to_tokens


def create_input_output_pairs(all_text_tensors, sequence_length=10, stride=3):
    # text = text.replace("\n", " ").strip()
    # tokens = text.split()  # Simple tokenization (split by whitespace)

    # input_output_pairs = []

    inputs, labels = [], []

    for i in range(0, len(all_text_tensors) - sequence_length, stride):
        inputs.append(all_text_tensors[i:i + sequence_length])
        labels.append(all_text_tensors[i + 1:i + sequence_length + 1])

    return inputs, labels


def data_loader(inputs, labels):
    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)

    # Create dataset and dataloader
    dataset = TensorDataset(inputs, labels)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    dataloader = DataLoader(dataset)
    return dataloader


def text_to_data(folder_path):
    text = read_text_files(folder_path)
    all_tokens_list = text_cleaning(text)
    all_tensors_list, vocab_size, tokens_to_idx, idx_to_tokens = tokens_to_tensors(all_tokens_list)

    inputs, labels = create_input_output_pairs(all_tensors_list)
    dataloader = data_loader(inputs, labels)
    return dataloader, vocab_size, tokens_to_idx, idx_to_tokens


