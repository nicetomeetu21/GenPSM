import json, os
import numpy as np

def save_metrics_to_json(test_metric_dict, global_iter, result_dir, json_name='single_fold_result.json'):
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    json_path = os.path.join(result_dir, json_name)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = dict()

    # print(results)
    def write_to_result(results_dict, iter, metric_dict):
        # print(iter, results_dict)
        iter = str(iter)
        if iter not in results_dict.keys(): results_dict[iter] = dict()
        # print(iter, results_dict)
        for k in metric_dict.keys():
            if k not in results_dict[iter]: results_dict[iter][k] = []
            results_dict[iter][k] += metric_dict[k]

    write_to_result(results, global_iter, test_metric_dict)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)


def save_metrics_to_json_v2(test_metric_list, global_iter, result_dir, json_name='results.json'):
    if not os.path.exists(result_dir): os.mkdir(result_dir)
    json_path = os.path.join(result_dir, json_name)
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            results = json.load(f)
    else:
        results = dict()

    # print(results)
    def write_to_result(results_dict, iter, metric_list):
        iter = str(iter)
        # assert iter not in results_dict.keys()
        results_dict[iter] = metric_list

    write_to_result(results, global_iter, test_metric_list)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

def get_mean_metrics(metrics_list):
    result = {}
    for metrics in metrics_list:
        for name in metrics.keys():
            if name == 'path': continue
            if name not in result.keys(): result[name] = []
            result[name].append(metrics[name])
    for name in result.keys():
        result[name] = np.mean(result[name])
    return result


def get_highest_and_lowest_id(metrics_list):
    result = {}
    for i, metrics in enumerate(metrics_list):
        for key in metrics.keys():
            if key == 'path': continue
            if key not in result.keys(): result[key] = {'high':i, 'low':i}
            if metrics[key] > metrics_list[result[key]['high']][key]: result[key]['high']=i
            if metrics[key] < metrics_list[result[key]['low']][key]: result[key]['low']=i
    return result
