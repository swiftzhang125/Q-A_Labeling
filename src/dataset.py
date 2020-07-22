import torch

class BERTDatasetTrain():
    def __init__(self, qtitle, qbody, qanswer, targets, tokenizer, max_len):
        self.qtitle = qtitle
        self.qbody = qbody
        self.qanswer = qanswer
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.qanswer)

    def __getitem__(self, item):
        question_title = str(self.qtitle[item])
        question_body = str(self.qbody[item])
        question_answer = str(self.qanswer[item])
        question_targets = self.targets[item, :]

        inputs = self.tokenizer.encode_plus(
            question_title + ' ' + question_body,
            question_answer,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len
        )

        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']

        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(question_targets, dtype=torch.float)
        }
