import torch
from monai.metrics import compute_dice
from monai.metrics import compute_hausdorff_distance
from monai.metrics import get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.utils import one_hot


def cal_metrics_monai(y_pred, y, num_classes, metric_dict):
    if 'dice' not in metric_dict: metric_dict['dice'] = []
    y_onehot = one_hot(y, num_classes=num_classes)
    y_pred_onehot = one_hot(y_pred, num_classes=num_classes)

    dices = compute_dice(y_pred_onehot, y_onehot)
    print(dices.shape)
    exit()
    dice = torch.mean(dices)
    metric_dict['dice'].append(dice.item())

    confusion_matrix = get_confusion_matrix(y_pred_onehot, y_onehot)
    sensitivitys = compute_confusion_matrix_metric("sensitivity", confusion_matrix)
    specificitys = compute_confusion_matrix_metric("specificity", confusion_matrix)
    precisions = compute_confusion_matrix_metric("precision", confusion_matrix)
    balanced_accuracys = compute_confusion_matrix_metric("balanced accuracy", confusion_matrix)
    accuracys = compute_confusion_matrix_metric("accuracy", confusion_matrix)
    f1_scores = compute_confusion_matrix_metric("f1 score", confusion_matrix)
    sensitivity = torch.mean(sensitivitys)
    specificity = torch.mean(specificitys)
    precision = torch.mean(precisions)
    balanced_accuracy = torch.mean(balanced_accuracys)
    accuracy = torch.mean(accuracys)
    f1_score = torch.mean(f1_scores)

    metric_dict['sensitivity'].append(sensitivity.item())
    metric_dict['specificity'].append(specificity.item())
    metric_dict['precision'].append(precision.item())
    metric_dict['balanced_accuracy'].append(balanced_accuracy.item())
    metric_dict['accuracy'].append(accuracy.item())
    metric_dict['f1_score'].append(f1_score.item())


def calculate_tp_tn_fp_fn(y_true, y_pred):
    """
    Calculate TP, TN, FP, FN for binary classification using PyTorch tensors.

    Args:
    - y_true: PyTorch tensor of shape (batch_size, 1) containing the true labels (0 or 1).
    - y_pred: PyTorch tensor of shape (batch_size, 1) containing the predicted labels (0 or 1).

    Returns:
    - tp: number of true positives.
    - tn: number of true negatives.
    - fp: number of false positives.
    - fn: number of false negatives.
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return tp, tn, fp, fn


def cal_metrics_multiclass(y_pred, y, path, num_classes, eval_metrics_list):
    # print(y_pred.shape, y.shape)
    y_pred = one_hot(y_pred, num_classes=num_classes)
    y = one_hot(y, num_classes=num_classes)
    result = {'path': path}
    for i in range(0, num_classes):
        y_pred_i = y_pred[:, i:i + 1, :, :]
        y_i = y[:, i:i + 1, :, :]
        if y_i.sum() == 0: continue

        TP, TN, FP, FN = calculate_tp_tn_fp_fn(y_i, y_pred_i)

        # miou = TP / (TP + FN + FP)
        # if f'{i}_miou' not in metric_dict: metric_dict[f'{i}_miou'] = []
        # metric_dict[f'{i}_miou'].append(miou.item())

        dice = 2 * TP / (2 * TP + FN + FP)
        result[f'{i}_dice'] = dice.item()
    eval_metrics_list.append(result)


def cal_metrics_v2(y_pred, y, path, eval_metrics_list, suffix):
    result = {'path': path}

    TP, TN, FP, FN = calculate_tp_tn_fp_fn(y, y_pred)

    tpr = TP / (TP + FN)
    tnr = TN / (FP + TN)
    dice = 2 * TP / (2 * TP + FN + FP)
    # print(y_pred.shape, y.shape)
    # print(TP, TN, FP, FN)

    # print(y_pred.shape, y.shape)
    # y_pred = one_hot(y_pred, num_classes=num_classes)
    # y = one_hot(y, num_classes=num_classes)
    hd = compute_hausdorff_distance(
        y_pred=y_pred, y=y, percentile=95
    )

    result[f'{suffix}/dice'] = dice.item()
    result[f'{suffix}/tpr'] = tpr.item()
    result[f'{suffix}/tnr'] = tnr.item()
    result[f'{suffix}/hd'] = hd.item()
    eval_metrics_list.append(result)



def cal_metrics_monai_v2(y_pred, y, num_classes, path, eval_metrics_list, suffix):
    result = {'path': path}
    y_onehot = one_hot(y, num_classes=num_classes)
    y_pred_onehot = one_hot(y_pred, num_classes=num_classes)

    dices = compute_dice(y_pred_onehot, y_onehot)

    dice = torch.mean(dices)
    result[f'{suffix}/dice'] = dice.item()
    eval_metrics_list.append(result)

def cal_metrics_monai_v3(y_pred, y, num_classes, path, eval_metrics_list, suffix):
    result = {'path': path}
    y_onehot = one_hot(y, num_classes=num_classes)
    y_pred_onehot = one_hot(y_pred, num_classes=num_classes)

    dices = compute_dice(y_pred_onehot, y_onehot)
    dices = dices[:,1:-1]
    # print(dices.shape)
    #
    # exit()
    dice = torch.mean(dices)
    result[f'{suffix}/dice'] = dice.cpu().item()
    eval_metrics_list.append(result)
def cal_metrics_monai_v4(y_pred, y, num_classes, path, eval_metrics_list, suffix):
    result = {'path': path}
    y_onehot = one_hot(y, num_classes=num_classes)
    y_pred_onehot = one_hot(y_pred, num_classes=num_classes)

    dices = compute_dice(y_pred_onehot, y_onehot)
    dices = dices[:,1:-1]
    # print(dices.shape)
    #
    # exit()
    for i in range(dices.shape[0]):
        dice = torch.mean(dices[i])

        result[f'{suffix}/dice'] = dice.cpu().item()
        eval_metrics_list.append(result)

def cal_hd(pred, label):
    pred = pred.unsqueeze(0)
    pred = (pred > 0.5).float()
    label = label.unsqueeze(0)
    return compute_hausdorff_distance(
        y_pred=pred, y=label, percentile=95
    )


def cal_masked_metrics(pred_out, pred_label, mask, metric_dict):
    if 'macc' not in metric_dict: metric_dict['macc'] = []
    if 'mtpr' not in metric_dict: metric_dict['mtpr'] = []
    if 'mtnr' not in metric_dict: metric_dict['mtnr'] = []
    if 'mmiou' not in metric_dict: metric_dict['mmiou'] = []
    if 'mdice' not in metric_dict: metric_dict['mdice'] = []
    if 'mhd' not in metric_dict: metric_dict['mhd'] = []

    TP = ((pred_out > 0.5) & (pred_label > 0.5))
    TN = ((pred_out <= 0.5) & (pred_label <= 0.5))
    FN = ((pred_out <= 0.5) & (pred_label > 0.5))
    FP = ((pred_out > 0.5) & (pred_label <= 0.5))
    TP = (TP.float() * mask).sum()
    TN = (TN.float() * mask).sum()
    FN = (FN.float() * mask).sum()
    FP = (FP.float() * mask).sum()
    # print(TP,TN, FN, FP)
    # print(mask.sum(), TP+TN+FN+FP)
    acc = (TP + TN) / (TP + FP + FN + TN)
    tpr = TP / (TP + FN)
    tnr = TN / (FP + TN)
    miou = TP / (TP + FN + FP)
    dice = 2 * TP / (2 * TP + FN + FP)

    metric_dict['macc'].append(acc.item())
    metric_dict['mtpr'].append(tpr.item())
    metric_dict['mtnr'].append(tnr.item())
    metric_dict['mmiou'].append(miou.item())
    metric_dict['mdice'].append(dice.item())
    # print(torch.max(mask))
    # mask //= 255
    hd = cal_hd(pred_out.clone() * mask, pred_label.clone() * mask)
    metric_dict['mhd'].append(hd.item())
