import json
import logging
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import find_start_end_before_tokenized
import torch.nn.functional as F
from data.utils import load_data
from data.hotpotqa import batch_transform_bert_hotpot
from data.fever import batch_transform_bert_fever

from data import HotpotDataset, FEVERDataset
from data import batcher_hotpot, batcher_fever
import itertools

def interpret_hotpot(model, config, args):
    pass

def interpret_fever(model, config, args):
    data = load_data(config["system"]['test_data'], False)
    inst = data[0]
    device = args.device
    collate_fn = batcher_fever(device)

    logging.info("=================================================================================")

    removed_ids = list(itertools.product(*[list(range(len(node["context"]) - 2)) for node in inst["node"]]))
    logs = {"inst": inst, "data": []}
    for k, ids in enumerate(removed_ids):
        removed_tokens = []
        tmp_inst = deepcopy(inst)
        for i, id_ in enumerate(ids):
            tokens_to_remove = []
            for j in range(3):
                tokens_to_remove.append(tmp_inst["node"][i]["context"][id_ + j])
            for j in range(3):
                del tmp_inst["node"][i]["context"][id_]
            removed_tokens.append(tokens_to_remove)

        batch = collate_fn([batch_transform_bert_fever(tmp_inst, config["model"]['bert_max_len'], args.tokenizer)])

        logits_score, logits_pred = model.network(batch, device)
        logits_score = F.softmax(logits_score)
        logits_pred = F.softmax(logits_pred, dim=1)
        final_score = torch.mm(logits_score.unsqueeze(0), logits_pred).squeeze(0)
        values, index = final_score.topk(1)
        label = batch[2]
        cls_, prob = index[0].item() == label[0].item(), values[0].item()

        logs["data"].append({"remove": removed_tokens, "label": cls_, "prob": prob})
        logging.info(logs["data"][-1])

    with open("results.json", "w") as f:
        json.dump(logs, f)
    return None