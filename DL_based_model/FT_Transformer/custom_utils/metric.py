import torch
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


def get_pred(predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return hard predictions and positive-class probability for binary classification."""
    logits = torch.cat([p.detach().cpu() for p in predictions], dim=0)  # [N, 2]
    probs = torch.softmax(logits, dim=1)                                # [N, 2]
    pred = probs.argmax(dim=1)                                          # [N]
    prob_pos = probs[:, 1]                                              # [N]
    return pred, prob_pos

def confusion(g_true: List[torch.Tensor], predictions: List[torch.Tensor]) -> Tuple[float, ...]:
    pred, prob = get_pred(predictions)

    y = torch.cat([t.detach().cpu() for t in g_true], dim=0)  # [N, 2]
    lab = y.argmax(dim=1)                                     # [N]

    lab_np = lab.numpy()
    pred_np = pred.numpy()
    prob_np = prob.numpy()

    acc = accuracy_score(lab_np, pred_np)
    tn, fp, fn, tp = confusion_matrix(lab_np, pred_np).ravel()

    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    auc = roc_auc_score(lab_np, prob_np)

    precision = tp / (tp + fp + 1e-8)
    recall = sens
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return acc, sens, spec, auc, f1, recall, precision
