import argparse
import os
# Define Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="KindMed")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--portion_train", type=float, default=2/3)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--ddi_target", type=float, default=0.08)
parser.add_argument("--Kp", type=float, default=0.05)
parser.add_argument("--with_meds", type=int, default=1)         # 0: False, 1: True
parser.add_argument("--with_fusion", type=int, default=1)       # 0: False, 1: True
parser.add_argument("--with_apm", type=int, default=1)          # 0: False, 1: True
parser.add_argument("--with_augment", type=int, default=1)      # 0: False, 1: True
parser.add_argument("--with_ontology", type=int, default=1)     # 0: False, 1: True
parser.add_argument("--with_semantics", type=int, default=1)    # 0: False, 1: True
parser.add_argument("--with_ddi", type=int, default=1)          # 0: False, 1: True
parser.add_argument("--with_demographics", type=int, default=1) # 0: False, 1: True
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--num_diagpro_relations", type=int, default=30)
parser.add_argument("--num_med_relations", type=int, default=3)
parser.add_argument("--weight_bce", type=float, default=0.95)
parser.add_argument("--weight_ddi_loss", type=float, default=0.0005)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--phase", type=str, default='training') # 'training', 'testing' (w/ Bootstrap sampling)
parser.add_argument("--note", type=str, default='KindMed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

import time
import warnings
def warn(*args, **kargs): pass
warnings.warn = warn
import torch
import torch_geometric as tg
import torch.optim as optim
import tensorflow as tf
import random
import datetime
from helpers import *
from tqdm import tqdm
from models import KindMed
from losses import DDILoss

# GPU Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device

# Seeding
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
tg.seed_everything(seed)

# Parameter Modification
if args.with_ontology == 0:
    args.num_diagpro_relations = args.num_diagpro_relations - 2
if args.with_semantics == 0:
    args.num_diagpro_relations = args.num_diagpro_relations - 26
if args.with_ontology == 0:
    args.num_med_relations = args.num_med_relations - 1
if args.with_ddi == 0:
    args.num_med_relations = args.num_med_relations - 1
if args.with_demographics == 0:
    args.num_diagpro_relations = args.num_diagpro_relations - 2
    args.num_med_relations = args.num_med_relations - 1

directory = None
if args.phase == 'training': # Training
    # Logging purpose for Training
    date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))
    directory = 'log/%s/%s_%s_GPU%s/' % (args.model, args.model, date_str, args.gpu_id)
    if not os.path.exists(directory):
        os.makedirs(directory)
        os.makedirs(directory + 'tflog/')
        os.makedirs(directory + 'model/')

    # Text Logging
    f = open(directory + 'setting.log', 'a')
    writelog(f, '======================')
    writelog(f, 'Model: %s' % args.model)
    writelog(f, 'Phase: %s' % args.phase)
    writelog(f, 'Learning Rate: %.5f' % args.lr)
    writelog(f, 'Epoch: %d' % args.epoch)
    writelog(f, 'Training Set Portion: %.5f' % args.portion_train)
    writelog(f, 'DIM: %d' % args.dim)
    writelog(f, 'DDI Target: %.5f' % args.ddi_target)
    writelog(f, 'Kp: %.5f' % args.Kp)
    writelog(f, 'With Medication KGs: %d' % args.with_meds)
    writelog(f, 'With Fusion: %d' % args.with_fusion)
    writelog(f, 'With APM: %d' % args.with_apm)
    writelog(f, 'With Augmentation: %d' % args.with_augment)
    writelog(f, 'With Ontology: %d' % args.with_ontology)
    writelog(f, 'With Semantics: %d' % args.with_semantics)
    writelog(f, 'With DDI: %d' % args.with_ddi)
    writelog(f, 'With Demographics: %d' % args.with_demographics)
    writelog(f, 'Number of Heads in MHA: %d' % args.num_heads)
    writelog(f, '# Clinical Relations: %d' % args.num_diagpro_relations)
    writelog(f, '# Medications Relations: %d' % args.num_med_relations)
    writelog(f, 'Weight BCE: %.5f' % args.weight_bce)
    writelog(f, 'Weight DDI Loss: %.5f' % args.weight_ddi_loss)
    writelog(f, 'Weight Decay: %.5f' % args.weight_decay)
    writelog(f, '----------------------')
    writelog(f, 'Note: %s' % args.note)
    writelog(f, '======================')
    f.close()
    f = open(directory + 'log_training.log', 'a')

    # Tensorboard Logging
    with tf.device('/cpu:0'):
        tfw_train = tf.summary.create_file_writer(directory + 'tflog/train_')
        tfw_valid = tf.summary.create_file_writer(directory + 'tflog/valid_')
else: # Testing
    path_to_model = 'KindMed_optimized'
    directory = 'log/%s/%s/' % (args.model, path_to_model)
    f = open(directory + 'log_testing.log', 'w')

def training(data, directory='.'):
    # Set mode as training
    model.train()

    # Define training variables
    loss_bce_ = 0
    loss_multi_ = 0
    loss_ddi_ = 0
    loss_total_ = 0
    counter_visits = 0

    # Loop over the patients and its corresponding visits
    for i, patient in enumerate(tqdm(data)):
        for t, visit in enumerate(patient):
            # Data
            g_c = [v['clinical_graph'].clone().detach().to(device) for v in patient[:t+1]]
            g_m = [v['medicine_graph'].clone().detach().to(device) for v in patient[:t+1]]
            y_m = [torch.LongTensor(list(v['med_label'])).to(device) for v in patient[:t+1]]
            y_binary = torch.zeros((1, voc_y_size)).to(device)
            y_binary[:, visit['med_label']] = 1
            y_multi = torch.full((1, voc_y_size), -1).to(device)
            y_multi[:, :len(visit['med_label'])] = torch.FloatTensor(visit['med_label']).to(device)

            # Forward Pass
            optimizer.zero_grad()
            y_pred_logit, y_pred_prob = model(g_c, g_m, y_m)
            loss_bce = criterion_bce(y_pred_logit, y_binary)
            loss_multi = criterion_multi(y_pred_prob, y_multi)
            loss_ddi = criterion_ddi(y_pred_prob, tensor_ddi_adj)

            y_pred = y_pred_prob.detach().cpu().numpy()[0]
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0
            y_pred = np.where(y_pred == 1)[0]
            ddi_rate_ = calculate_ddi_rate_score([[y_pred]], ddi_adj)
            if ddi_rate_ <= args.ddi_target:
                # since ddi_rate_ already low, just use bce+multi
                loss_total = args.weight_bce * loss_bce + (1.0-args.weight_bce) * loss_multi
            else:
                # Apply proportional-integral-derivative (PID) controller
                lambda_ddi = min(0, 1 + (args.ddi_target - ddi_rate_) / args.Kp)
                loss_total = lambda_ddi * (args.weight_bce * loss_bce + (1.0-args.weight_bce) * loss_multi) + (1-lambda_ddi) * loss_ddi
            loss_total.backward(retain_graph=True)
            optimizer.step()

            # For logging
            loss_bce_ += args.weight_bce * loss_bce.item()
            loss_multi_ += (1.0-args.weight_bce) * loss_multi.item()
            loss_ddi_ += loss_ddi.item()
            loss_total_ += loss_total.item()
            counter_visits += 1

    # Take average
    loss_bce_ = loss_bce_ / counter_visits
    loss_multi_ = loss_multi_ / counter_visits
    loss_ddi_ = loss_ddi_ / counter_visits
    loss_total_ = loss_total_ / counter_visits
    writelog(f, 'Loss Binary: %.8f' % loss_bce_)
    writelog(f, 'Loss Multi: %.8f' % loss_multi_)
    writelog(f, 'Loss DDI: %.8f' % loss_ddi_)
    writelog(f, 'Loss Total: %.8f' % loss_total_)

    # Tensorboard Logging
    info = {'loss_bce': loss_bce_,
            'loss_multi': loss_multi_,
            'loss_ddi': loss_ddi_,
            'loss_total': loss_total_}

    for tag, value in info.items():
        with tfw_train.as_default():
            with tf.device('/cpu:0'):
                tf.summary.scalar(tag, value, step=epoch)

def evaluate(phase, data, directory='.'):
    # Set mode as eval
    model.eval()

    # Define evaluation variables
    all_y_hat_labels = []
    metric_jaccard, metric_prauc, metric_prec, metric_recall, metric_f1 = [[] for _ in range(5)]
    counter_visits, counter_meds = 0, 0

    # # No Grad
    with torch.no_grad():
        # Loop over the patients and its corresponding visits
        for i, patient in enumerate(tqdm(data)):
            y_gts, y_hat_preds, y_hat_probs, y_hat_labels = [], [], [], []
            for t, visit in enumerate(patient):
                # Data
                g_c = [v['clinical_graph'].clone().detach().to(device) for v in patient[:t+1]]
                g_m = [v['medicine_graph'].clone().detach().to(device) for v in patient[:t+1]]
                y_m = [torch.LongTensor(list(v['med_label'])).to(device) for v in patient[:t+1]]

                # Forward
                y_pred_logit, y_pred_prob = model(g_c, g_m, y_m)
                y_pred_prob = y_pred_prob.squeeze(0).detach().cpu().numpy()

                # For logging, Collect all required variables for measuring performances
                y_gt = np.zeros(voc_y_size)
                y_gt[visit['med_label']] = 1
                y_gts.append(y_gt)
                y_hat_probs.append(y_pred_prob)
                y_hat_pred_ = y_pred_prob.copy()
                y_hat_pred_[y_hat_pred_ >= 0.5] = 1
                y_hat_pred_[y_hat_pred_ < 0.5] = 0
                y_hat_preds.append(y_hat_pred_)
                y_hat_label_ = np.where(y_hat_pred_ == 1)[0]
                y_hat_labels.append(sorted(y_hat_label_))
                counter_meds += len(y_hat_label_)
                counter_visits += 1

            # Classification Performance per Visits
            jaccard_, prauc_, prec_, recall_, f1_ = calculate_performance(np.array(y_gts),
                                                                          np.array(y_hat_preds),
                                                                          np.array(y_hat_probs))
            metric_jaccard.append(jaccard_)
            metric_prauc.append(prauc_)
            metric_prec.append(prec_)
            metric_recall.append(recall_)
            metric_f1.append(f1_)
            all_y_hat_labels.append(y_hat_labels)

        # Tensorboard Logging
        # Classification Performance
        metric_ddi_rate = calculate_ddi_rate_score(all_y_hat_labels, ddi_adj)
        metric_meds = counter_meds / counter_visits
        metric_jaccard = np.mean(metric_jaccard)
        metric_prauc = np.mean(metric_prauc)
        metric_prec = np.mean(metric_prec)
        metric_recall = np.mean(metric_recall)
        metric_f1 = np.mean(metric_f1)
        writelog(f, 'DDI-RATE: %.4f' % (metric_ddi_rate))
        writelog(f, 'JACCARD: %.4f' % (metric_jaccard))
        writelog(f, 'PRAUC: %.4f' % (metric_prauc))
        writelog(f, 'PREC: %.4f' % (metric_prec))
        writelog(f, 'RECALL: %.4f' % (metric_recall))
        writelog(f, 'F1-SCORE: %.4f' % (metric_f1))
        writelog(f, 'MEDS: %.4f' % (metric_meds))
        info = {'metric_ddi_rate': metric_ddi_rate,
                'metric_jaccard': metric_jaccard,
                'metric_prauc': metric_prauc,
                'metric_prec': metric_prec,
                'metric_recall': metric_recall,
                'metric_f1': metric_f1,
                'metric_meds': metric_meds}

        if phase=='validation':
            for tag, value in info.items():
                with tfw_valid.as_default():
                    with tf.device('/cpu:0'):
                        tf.summary.scalar(tag, value, step=epoch)

        return metric_ddi_rate, metric_jaccard, metric_prauc, metric_prec, metric_recall, metric_f1, metric_meds

def testing(phase, data_test, directory='.'):
    # Set mode as eval
    model.eval()
    # Performances
    performances = {'DDI-RATE': [], 'JACCARD':[], 'PRAUC':[], 'PREC':[], 'RECALL': [], 'F1-SCORE': [], 'MEDS': []}
    test_times = []
    # Do bootstrap sampling for 10 rounds
    for test_i in range(10):
        tic = time.time()
        writelog(f, 'Test #%d' % test_i)
        data_test_ = np.random.choice(
            np.array(data_test, dtype=object), round(len(data_test) * 0.8), replace=True
        )
        metric_ddi_rate, metric_jaccard, metric_prauc, metric_prec, metric_recall, metric_f1, metric_meds = evaluate('testing', data_test_, directory=directory)
        performances['DDI-RATE'].append(metric_ddi_rate)
        performances['JACCARD'].append(metric_jaccard)
        performances['PRAUC'].append(metric_prauc)
        performances['PREC'].append(metric_prec)
        performances['RECALL'].append(metric_recall)
        performances['F1-SCORE'].append(metric_f1)
        performances['MEDS'].append(metric_meds)
        times_ = time.time() - tic
        test_times.append(times_)
        writelog(f, 'Testing Time %.3f' % (times_))
        writelog(f, '---')
    writelog(f, 'Avg Testing Time %.3f' % (np.array(test_times).mean()))
    writelog(f, '---')
    for k, values in performances.items():
        writelog(f, '%s: %.4f \pm %.4f' % (k, np.mean(values), np.std(values)))
    writelog(f, '---')
    for k, values in performances.items():
        writelog(f, '%s: %s' % (k, ','.join([str(v) for v in values])))

# Define Loader
writelog(f, 'Load data')
records, ehr_adj, ddi_adj, ddi_mask, _, voc = data_loader(load_graph=True)
data_train, data_valid, data_test = data_splitter(records, args)
voc_d, voc_p, voc_m, voc_y = voc["diag_voc"], \
                             voc["pro_voc"], \
                             voc["med_voc"], \
                             voc["med_label_voc"]
voc_d_size, voc_p_size, voc_m_size, voc_y_size = len(voc_d.idx2word), \
                                                 len(voc_p.idx2word), \
                                                 len(voc_m.idx2word), \
                                                 len(voc_y.idx2word)
voc_size = (voc_d_size, voc_p_size, voc_m_size, voc_y_size)
tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)

# Define Models, Losses, and Optimizers
model = KindMed(args, voc_size, device).to(device)
if args.phase == 'testing': # Load trained model parameters for Testing
    model.load_state_dict(torch.load(directory + 'model/' + args.model + '.pt'))
criterion_bce = torch.nn.BCEWithLogitsLoss().to(device)
criterion_multi = torch.nn.MultiLabelMarginLoss().to(device)
criterion_ddi = DDILoss(args).to(device)
params = list(model.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
writelog(f, 'Total params is {}'.format(total_params))

# Best epoch checking
valid = {'epoch': 0, 'jaccard': 0}

if args.phase == 'training':
    # Training Epoch
    for epoch in range(args.epoch):
        writelog(f, '--- Epoch %d' % epoch)
        tic = time.time()
        writelog(f, 'Training')
        training(data_train, directory=directory)
        times_ = time.time() - tic
        writelog(f, 'Training Time %.3f' % (times_))

        tic = time.time()
        writelog(f, 'Validation')
        _, metric_jaccard, _, _, _, _, _ = evaluate('validation', data_valid, directory=directory)
        times_ = time.time() - tic
        writelog(f, 'Validation Time %.3f' % (times_))

        # Save Model
        if metric_jaccard >= valid['jaccard']:
            torch.save(model.state_dict(), directory + 'model/' + args.model + '.pt')
            writelog(f, 'Best Jaccard is found! Jaccard : %f' % metric_jaccard)
            writelog(f, 'Models at Epoch %d are saved!' % epoch)
            valid['epoch'] = epoch
            valid['jaccard'] = metric_jaccard
            f2 = open(directory + 'setting.log', 'a')
            writelog(f2, 'Best Epoch %d' % epoch)
            f2.close()
    writelog(f, 'Best model for testing: epoch %d-th' % valid['epoch'])

else:
    writelog(f, 'Testing')
    testing('testing', data_test, directory=directory)

writelog(f, 'END OF %s PROCESS' % str(args.phase).upper())
f.close()
