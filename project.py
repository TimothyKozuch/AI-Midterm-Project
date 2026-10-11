import re
import json
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

N_PROMPTS = 1000    #how many prompts to use, every prompt gives one row per LLM
MAX_LEN = 200       #every response is cut or padded to this many tokens
INPUT_LEN = 50      #in the input + output setting the prompt is cut to this many tokens
EPOCHS = 8
SEEDS = [0, 1, 2]   #every experiment is repeated with each seed, then averaged

#load dataset
#the files in data/ come from github.com/locuslab/llm-idiosyncrasies and were cleaned so that
#all three files have the same prompts in the same order (conversation number i has the same prompt in every file)
#each file is a list of conversations: [{"role": "user", ...prompt}, {"role": "assistant", ...response}]

llmNames = ["ChatGPT", "Claude", "Gemini"]

conversations = {}
for llmName in llmNames:
    conversations[llmName] = json.load(open(f"data/{llmName}.json"))

#turn text into a list of tokens
#words, punctuation and new lines are all separate tokens

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]|\n", text.lower())

#make one row for every (LLM, prompt, response)

labels = []         #0, 1 or 2: which LLM wrote the response
promptIds = []      #which prompt the row belongs to
inputTokens = []
outputTokens = []
for promptId in range(N_PROMPTS):
    for label, llmName in enumerate(llmNames): # enumerate gives num and value from list ex: (0,"ChatGPT")
        conversation = conversations[llmName][promptId]
        labels.append(label) #results in 0,1,2,0,1,2,0,1,2,...
        promptIds.append(promptId) #results in 0,0,0,1,1,1,2,2,2...
        inputTokens.append(tokenize(conversation[0]["content"])) # come back to this
        outputTokens.append(tokenize(conversation[1]["content"]))
labels = np.array(labels)
promptIds = np.array(promptIds)

#the third setting: the start of the prompt up to the INPUT_LEN, a separator, then the response
inputAndOutputTokens = [
    i[:INPUT_LEN] + ["<sep>"] + o 
    for i, o in zip(inputTokens, outputTokens)
]

#split the data into a train set and a test set
#all responses to the same prompt stay together, so test prompts are never seen in training

def splitData(seed):
    promptsTrain, promptsTest = train_test_split(np.arange(N_PROMPTS), test_size=0.25, random_state=seed)
    idxTrain = np.where(np.isin(promptIds, promptsTrain))[0]
    idxTest = np.where(np.isin(promptIds, promptsTest))[0]
    return idxTrain, idxTest

#why not like this #split the data into a train set and a test set
# weatherTrain, weatherTest, seaStateTrain, seaStateTest = train_test_split(
#     weather, seaState,test_size=0.25, random_state=RANDOM_SEED, stratify=seaState
# )

#the two required models
#padding_idx=0 means the padding tokens are all zeros and are ignored

class CNN(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.embed = nn.Embedding(vocabSize, 32, padding_idx=0)   #every token becomes 32 numbers
        self.conv = nn.Conv1d(32, 64, kernel_size=3)              #looks at 3 tokens at a time
        self.out = nn.Linear(64, len(llmNames))

    def forward(self, x):
        x = self.embed(x).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = x.max(dim=2).values   #keep the strongest signal from the whole text
        return self.out(x)

class RNN(nn.Module):
    def __init__(self, vocabSize):
        super().__init__()
        self.embed = nn.Embedding(vocabSize, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.out = nn.Linear(64, len(llmNames))

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embed(x))
        return self.out(hidden[0])   #the LSTM's memory after reading the whole text

#train one model once on the train set and predict the test set

def trainAndPredict(modelName, tokenLists, maxLen, seed):
    torch.manual_seed(seed)
    idxTrain, idxTest = splitData(seed)
    tokensTrain = [tokenLists[i] for i in idxTrain]
    tokensTest = [tokenLists[i] for i in idxTest]

    #give every token that shows up at least twice in the train set a number
    counts = Counter(token for tokens in tokensTrain for token in tokens)
    vocab = {"<pad>": 0, "<unknown>": 1}
    for token, count in counts.items():
        if count >= 2:
            vocab[token] = len(vocab)

    #turn every text into a row of numbers, padded at the front so the LSTM ends on real tokens
    def encode(tokenLists):
        numbers = np.zeros((len(tokenLists), maxLen), dtype=np.int64)
        for i, tokens in enumerate(tokenLists):
            ids = [vocab.get(token, 1) for token in tokens][:maxLen]
            numbers[i, maxLen - len(ids):] = ids
        return torch.tensor(numbers)

    xTrain = encode(tokensTrain)
    xTest = encode(tokensTest)
    yTrain = torch.tensor(labels[idxTrain])

    if modelName == "CNN":
        model = CNN(len(vocab))
    else:
        model = RNN(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lossFunction = nn.CrossEntropyLoss()

    epochLosses = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(xTrain))   #shuffle, then train in batches of 32
        totalLoss = 0
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            optimizer.zero_grad()
            loss = lossFunction(model(xTrain[batch]), yTrain[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   #stops the LSTM training from blowing up
            optimizer.step()
            totalLoss += loss.item() * len(batch)
        epochLosses.append(round(totalLoss / len(order), 2))
    print("   average train loss per epoch:", epochLosses)

    with torch.no_grad():
        return model(xTest).argmax(dim=1).numpy(), labels[idxTest]

#repeat with every seed, then print the average accuracy

def runExperiment(modelName, tokenLists, maxLen=MAX_LEN):
    accuracies = []
    allMatrices = np.zeros((len(llmNames), len(llmNames)), dtype=int)
    for seed in SEEDS:
        prediction, correct = trainAndPredict(modelName, tokenLists, maxLen, seed)
        accuracies.append(np.mean(prediction == correct))
        allMatrices += confusion_matrix(correct, prediction, labels=range(len(llmNames)))

    #shows which LLMs got predicted correctly and which did not (added up over all seeds)
    print(allMatrices)
    print(f"Test accuracy: {np.mean(accuracies):.3f} +/- {np.std(accuracies):.3f}")
    return np.mean(accuracies), np.std(accuracies)

#RQ1 and RQ2: every model on three different inputs

settings = {
    "Output only (RQ1)": (outputTokens, MAX_LEN),
    "Input only (RQ2)": (inputTokens, MAX_LEN),
    "Input + output (RQ2)": (inputAndOutputTokens, INPUT_LEN + 1 + MAX_LEN),
}

results = {}
for settingName, (tokenLists, maxLen) in settings.items():
    for modelName in ["CNN", "RNN"]:
        print(f"\n{modelName}, {settingName}")
        results[(settingName, modelName)] = runExperiment(modelName, tokenLists, maxLen)

#summary of everything

print("\n\nSummary (chance level =", round(1 / len(llmNames), 3), ")")
for (settingName, modelName), (mean, std) in results.items():
    print(f"{settingName:45s} {modelName:9s} {mean:.3f} +/- {std:.3f}")