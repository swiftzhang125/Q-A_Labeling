import torch.nn as nn
import os
import torch
import numpy as np
import pandas as pd
import transformers
from model_dispatcher import MODEL_DISPATCHER
from dataset import BERTDatasetTrain
from sklearn import model_selection
from scipy import stats
from transformers import AdamW, get_linear_schedule_with_warmup
import warnings

warnings.filterwarnings('ignore')

DEVICE = 'cuda'

MAX_LEN = int(os.environ.get('MAX_LEN'))
EPOCHS = int(os.environ.get('EPOCHS'))
TRAIN_BATCH_SIZE = int(os.environ.get('TRAIN_BATCH_SIZE'))
VALIDATION_BATCH_SIZE = int(os.environ.get('VALIDATION_BATCH_SIZE'))
LR = float(os.environ.get('LR'))

BASE_MODEL = os.environ.get('BASE_MODEL')
BERT_PATH = os.environ.get('BERT_PATH')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH')

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_loop_fn(dataloader, model, optimizer, device, scheduler=None):
    model.train()
    for step, d in enumerate(dataloader):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if step % 10 == 0 and step != 0:
            print(f'step = {step}, loss = {loss}')


def eval_loop_fn(dataloader, model, device):
    model.eval()
    fin_o = []
    fin_t = []
    for step, d in enumerate(dataloader):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        fin_o.append(outputs.cpu().detach().numpy())
        fin_t.append(targets.cpu().detach().numpy())

    return np.vsplit(fin_o), np.vstack(fin_t)


def run():
    df = pd.read_csv('../input/google-quest-challenge/train.csv').fillna('none')
    df_train, df_valid = model_selection.train_test_split(df, random_state=2020, test_size=0.1)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    sample = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')
    target_cols = list(sample.drop('qa_id', axis=1).columns)

    train_targets = df_train[target_cols].values
    valid_targets = df_valid[target_cols].values

    tokenizer = transformers.BertTokenizer.from_pretrained(TOKENIZER_PATH)

    train_dataset = BERTDatasetTrain(
        qtitle=df_train.question_title.values,
        qbody=df_train.question_body.values,
        qanswer=df_train.answer.values,
        targets=train_targets,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    valid_dataset = BERTDatasetTrain(
        qtitle=df_valid.question_title.values,
        qbody=df_valid.question_body.values,
        qanswer=df_valid.answer.values,
        targets=valid_targets,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=VALIDATION_BATCH_SIZE,
        shuffle=False
    )

    model = MODEL_DISPATCHER['bert'](bert_path='bert-base-uncased')
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = int(len(train_dataset) * TRAIN_BATCH_SIZE * EPOCHS)

    sheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(EPOCHS):
        train_loop_fn(train_dataloader, model, optimizer, DEVICE, scheduler=sheduler)
        o, t = eval_loop_fn(valid_dataloader, model, DEVICE)

        coef = []
        for v in range(t.shape[1]):
            p1 = list(t[:, v])
            p2 = list(o[:, v])

            rho, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            coef.append(rho)

        coef = np.mean(coef)
        print(f'epoch = {epoch}, coef = {coef}')


if __name__ == '__main__':
    run()





