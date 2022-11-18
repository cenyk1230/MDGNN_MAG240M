import sys
import os
import glob
import numpy as np
import torch

from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from features.config_feats import model_feats


dataset_path, input_path, output_path, feat_path = sys.argv[1:5]

jax_feat_path = os.path.join(feat_path, "x_jax_153.npy")

models = [list(x.keys())[0] for x in model_feats]
weights = [
    -0.4877865744749357, 
    -0.346649996233688, 
    -0.37787874613394634, 
    -0.5604140184991806, 
    0.8799434501215753, 
    0.46518827818407305, 
    -0.3769751431472341, 
    0.8269823445831963, 
    -0.1954645260389538, 
    0.1888043427524997, 
    0.6469049538984475, 
    -0.2883804174068405, 
    -0.25362940422525115, 
    0.48079288659829533, 
    0.8627485259401018
]

dataset = MAG240MDataset(root=dataset_path)
split_nids = dataset.get_idx_split()
node_ids = np.concatenate([split_nids['train'], split_nids['valid']])

y_true_train = dataset.paper_label[dataset.get_idx_split('train')]
y_true_valid = dataset.paper_label[dataset.get_idx_split('valid')]
y_all = np.concatenate([y_true_train, y_true_valid])
evaluator = MAG240MEvaluator()

jax_feat = np.load(jax_feat_path)
y_pred_val_jax = jax_feat[len(y_true_train):len(y_true_train) + len(y_true_valid)]
y_pred_test_jax = jax_feat[len(y_true_train) + len(y_true_valid):]

method_weight = [(x[0],weights[ii]) for ii,x in enumerate(models)]

print("\nEvaluating at validation dataset...")
y_pred_valid_all = []
acc_all = []
for ii, (method, weight) in enumerate(method_weight):
    method_path = os.path.join(input_path, method)
    y_pred_v = []
    idx_v = []
    # print("\n %d %s..." % (ii,method))
    for fpath in glob.glob(os.path.join(method_path, 'cv-*')):
        # print("Loading predictions from %s" % fpath)
        y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_valid.npy"))).softmax(axis=1).numpy()
        y_pred_v.append(y)
        idx = np.load(os.path.join(fpath, "idx_valid.npy"))
        idx_v.append(idx)
    idx_v = np.concatenate(idx_v, axis=0)
    y_pred_v = np.concatenate(y_pred_v, axis=0)
    y_true_v = y_all[idx_v]
    y_pred_valid = y_pred_v[np.argsort(idx_v)]
    y_pred_valid_all.append(y_pred_valid * weight)
    np.save(os.path.join(method_path, 'y_pred_valid_all.npy'), y_pred_valid)

    acc = evaluator.eval(
        {'y_true': y_true_v, 'y_pred': y_pred_v.argmax(axis=1)}
    )['acc']
    acc_all.append(acc)
    # print("valid accurate: %.4f" % acc)

# add jax feats
weight_jax = weights[-1]
y_pred_valid_all.append(y_pred_val_jax * weight_jax)

nsample, ndim = y_pred_valid_all[0].shape
y_pred_valid_all = np.concatenate(y_pred_valid_all).reshape((-1, nsample, ndim)).sum(axis=0)
acc = evaluator.eval(
    {'y_true': y_true_valid, 'y_pred': y_pred_valid_all.argmax(axis=1)}
)['acc']
print("valid ensemble (average) accurate: %.4f" % acc)

# process tests
y_pred_test_all = []
for ii, (method, weight) in enumerate(method_weight):
    method_path = os.path.join(input_path, method)
    y_pred = []
    for fpath in glob.glob(os.path.join(method_path, 'cv-*')):
        y = torch.as_tensor(np.load(os.path.join(fpath, "y_pred_test.npy"))).softmax(axis=1).numpy()
        y_pred.append(y)
    nsample, ndim = y_pred[0].shape
    y_pred = np.concatenate(y_pred).reshape((-1, nsample, ndim)).mean(axis=0)
    y_pred_test_all.append(y_pred * weight)

y_pred_test_all.append(y_pred_test_jax * weight_jax)

y_pred_test = np.concatenate(y_pred_test_all).reshape((-1, nsample, ndim)).sum(axis=0)
res = {'y_pred': y_pred_test.argmax(axis=1)}

# process test-challenge
test_idx = split_nids['test-whole']
# print(test_idx.shape)
test_challenge_idx = split_nids['test-challenge']
size = int(test_idx.max()) + 1
test_challenge_mask = torch.zeros(size, dtype=torch.bool)
test_challenge_mask[test_challenge_idx] = True
test_challenge_mask = test_challenge_mask[test_idx]

res_challenge = {}
res_challenge['y_pred'] = res['y_pred'][test_challenge_mask]

print("Saving challenge to %s" % output_path)
evaluator.save_test_submission(res_challenge, output_path, mode="test-challenge")

#process test-dev
res_dev = {}
test_dev_mask = ~test_challenge_mask
res_dev['y_pred'] = res['y_pred'][test_dev_mask]
print("Saving dev to %s" % output_path)
evaluator.save_test_submission(res_dev, output_path, mode="test-dev")
