import models

MODEL_DISPATCHER = {
    'bert': models.BERTBaseUncased
}

if __name__ == '__main__':
    print(MODEL_DISPATCHER['bert'](bert_path='bert-base-uncased'))