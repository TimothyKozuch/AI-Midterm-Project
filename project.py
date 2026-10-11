import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

N_PROMPTS = 1000
MAX_LEN = 500
INPUT_LEN = 100
EPOCHS = 8
SEED = 42

llmNames = ["ChatGPT", "Claude", "Gemini"]

conversations = {}
for llmName in llmNames:
    with open(f"data/{llmName}.json", encoding="utf-8") as file:
        conversations[llmName] = json.load(file)

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]|\n", text.lower())

inputTokens, outputTokens = [], []
for promptId in range(N_PROMPTS):
    for llmName in llmNames:
        user, assistant = conversations[llmName][promptId]
        inputTokens.append(tokenize(user["content"]))
        outputTokens.append(tokenize(assistant["content"]))

labels = np.tile(np.arange(len(llmNames)), N_PROMPTS)
inputAndOutputTokens = [
    prompt[:INPUT_LEN] + ["<sep>"] + output
    for prompt, output in zip(inputTokens, outputTokens)
]

def splitData():
    trainPrompts, testPrompts = train_test_split(
        np.arange(N_PROMPTS), test_size=0.25, random_state=SEED
    )
    rows = np.arange(N_PROMPTS * len(llmNames)).reshape(N_PROMPTS, len(llmNames))
    return rows[trainPrompts].flatten(), rows[testPrompts].flatten()

class CNN(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.embed = nn.Embedding(vocabSize, 32, padding_idx=0)
        self.conv = nn.Conv1d(32, 64, kernel_size=3)
        self.out = nn.Linear(64, len(llmNames))

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = x.max(dim=2).values
        return self.out(x)

class RNN(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.embed = nn.Embedding(vocabSize, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.out = nn.Linear(64, len(llmNames))

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embed(x))
        return self.out(hidden[0])

def trainAndEvaluate(modelName, tokenLists, maxLen):
    torch.manual_seed(SEED)
    idxTrain, idxTest = splitData()
    tokensTrain = [tokenLists[i] for i in idxTrain]
    tokensTest = [tokenLists[i] for i in idxTest]

    counts = Counter(token for tokens in tokensTrain for token in tokens)
    vocab = {"<pad>": 0, "<unknown>": 1}
    for token, count in counts.items():
        if count >= 2:
            vocab[token] = len(vocab)

    def encode(tokenLists):
        numbers = np.zeros((len(tokenLists), maxLen), dtype=np.int64)
        for i, tokens in enumerate(tokenLists):
            ids = [vocab.get(token, 1) for token in tokens][:maxLen]
            numbers[i, maxLen - len(ids):] = ids
        return torch.tensor(numbers)

    xTrain = encode(tokensTrain)
    xTest = encode(tokensTest)
    yTrain = torch.tensor(labels[idxTrain])
    model = {"CNN": CNN, "RNN": RNN}[modelName](len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFunction = nn.CrossEntropyLoss()
    trainLoader = DataLoader(
        TensorDataset(xTrain, yTrain), batch_size=32, shuffle=True
    )

    for _ in range(EPOCHS):
        for x, y in trainLoader:
            optimizer.zero_grad()
            loss = lossFunction(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(xTest).argmax(dim=1).numpy()
    accuracy = np.mean(predictions == labels[idxTest])
    print(f"Test accuracy: {accuracy:.3f}")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(labels[idxTest], predictions, labels=range(len(llmNames))))

settings = {
    "Output only": (outputTokens, MAX_LEN),
    "Input only": (inputTokens, MAX_LEN),
    "Input + output": (inputAndOutputTokens, INPUT_LEN + 1 + MAX_LEN),
}

for settingName, (tokenLists, maxLen) in settings.items():
    for modelName in ("CNN", "RNN"):
        print(f"\n{modelName}, {settingName}")
        trainAndEvaluate(modelName, tokenLists, maxLen)