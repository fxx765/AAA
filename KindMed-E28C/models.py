import torch
import torch.nn as nn
import torch_geometric.nn as tgnn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from helpers import augment_graph

class NeuCF(nn.Module):
    def __init__(self, args):
        super(NeuCF, self).__init__()
        self.args = args
        self.layers = [2, 2, 2]
        self.relu = nn.ReLU()
        self.f_conv_l0 = nn.Conv1d(2, 1, kernel_size=1, stride=1)
        self.f_mlp_l1 = nn.ModuleList(
            [nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU()) for i in range(self.layers[0])]
        )
        self.f_conv_l1 = nn.Conv1d(self.layers[0], 1, kernel_size=1, stride=1)

        self.f_mlp_l2= nn.ModuleList(
            [nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU()) for i in range(self.layers[1])]
        )
        self.f_conv_l2 = nn.Conv1d(self.layers[1], 1, kernel_size=1, stride=1)

        self.f_mlp_l3 = nn.ModuleList(
            [nn.Sequential(nn.Linear(args.dim, args.dim), nn.ReLU()) for i in range(self.layers[2])]
        )
        self.f_conv_l3 = nn.Conv1d(self.layers[2], 1, kernel_size=1, stride=1)

        self.f_merge = nn.Sequential(
            nn.Linear(args.dim*2, args.dim*2),
            nn.ReLU(),
            nn.Linear(args.dim*2, args.dim*2),
        )
        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x_1, x_2):
        # GMF
        x_12_gmf = x_1 * x_2
        # DeepFusion
        x_12_mlp = self.relu(self.f_conv_l0(torch.cat([x_1, x_2], dim=0)))
        x_12_mlp = self.relu(self.f_conv_l1(torch.cat([self.f_mlp_l1[i](x_12_mlp) for i in range(self.layers[0])], dim=0)))
        x_12_mlp = self.relu(self.f_conv_l2(torch.cat([self.f_mlp_l2[i](x_12_mlp) for i in range(self.layers[1])], dim=0)))
        x_12_mlp = self.relu(self.f_conv_l3(torch.cat([self.f_mlp_l3[i](x_12_mlp) for i in range(self.layers[2])], dim=0)))
        # Merge
        x_12 = self.f_merge(torch.cat([x_12_gmf, x_12_mlp], dim=1))
        return x_12

# KindMed
class KindMed(nn.Module):
    def __init__(self, args, voc_size, device):
        super(KindMed, self).__init__()
        self.args = args
        self.voc_size = voc_size
        self.device = device
        # For initial embedding
        self.f_embed_dict = nn.ModuleDict()
        if args.with_demographics == 1:
            self.f_embed_dict['patient_gender'] = nn.Embedding(2, args.dim)
            self.f_embed_dict['patient_ethnicity'] = nn.Embedding(7, args.dim)
            self.f_embed_dict['patient_age'] = nn.Embedding(17, args.dim)
        self.f_embed_dict['diag'] = nn.Embedding(voc_size[0], args.dim)
        self.f_embed_dict['pro'] = nn.Embedding(voc_size[1], args.dim)

        # For medical KGs representation learning
        self.num_diagpro_relations = args.num_diagpro_relations
        self.c_conv1 = tgnn.FastRGCNConv(args.dim, args.dim, num_relations=self.num_diagpro_relations)
        self.c_conv2 = tgnn.FastRGCNConv(args.dim, args.dim, num_relations=self.num_diagpro_relations)
        if args.with_meds == 1:
            self.f_embed_dict['med'] = nn.Embedding(voc_size[2], args.dim)
            self.num_med_relations = args.num_med_relations
            self.m_conv1 = tgnn.FastRGCNConv(args.dim, args.dim, num_relations=self.num_med_relations)
            self.m_conv2 = tgnn.FastRGCNConv(args.dim, args.dim, num_relations=self.num_med_relations)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.readout = tgnn.aggr.SumAggregation()

        # For temporal learning
        self.c_rnn = nn.GRUCell(args.dim, args.dim)
        if args.with_meds == 1:
            if args.with_fusion == 1:
                self.neucf = NeuCF(args)
            self.m_rnn = nn.GRUCell(args.dim, args.dim)
            self.cm_rnn = nn.GRUCell(args.dim*2, args.dim)

        # For recommending a set of medicines
        if args.with_apm == 1:
            self.mha_triad = torch.nn.MultiheadAttention(args.dim, args.num_heads)
            self.norm = nn.LayerNorm(args.dim)
        else:
            if args.with_meds == 1:
                self.merge = nn.Sequential(
                    nn.Linear(args.dim*2, args.dim),
                    nn.ReLU(),
                    nn.Linear(args.dim, args.dim),
                )
                self.norm = nn.LayerNorm(args.dim)
        self.f_out = nn.Sequential(
            nn.Linear(args.dim, args.dim*2),
            nn.ReLU(),
            nn.Linear(args.dim*2, voc_size[3])
        )
        self.init_weights()

    def init_weights(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

        initrange = 0.1
        if self.args.with_demographics == 1:
            self.f_embed_dict['patient_gender'].weight.data.uniform_(-initrange, initrange)
            self.f_embed_dict['patient_ethnicity'].weight.data.uniform_(-initrange, initrange)
            self.f_embed_dict['patient_age'].weight.data.uniform_(-initrange, initrange)
        self.f_embed_dict['diag'].weight.data.uniform_(-initrange, initrange)
        self.f_embed_dict['pro'].weight.data.uniform_(-initrange, initrange)
        if self.args.with_meds == 1:
            self.f_embed_dict['med'].weight.data.uniform_(-initrange, initrange)

    def mkg_remove_ddi(self, g):
        del g['med', 'ddi', 'med']
        return g

    def mkg_remove_ontology(self, g):
        del g['med', 'atcontology', 'med']
        return g

    def mkg_remove_demographics(self, g):
        del g['med', 'belongsto', 'patient']
        del g['patient']
        return g

    def ckg_remove_semantics(self, g):
        del g['diag', 'affects', 'diag']
        del g['pro', 'affects', 'diag']
        del g['pro', 'affects', 'pro']
        del g['diag', 'associatedwith', 'diag']
        del g['diag', 'augments', 'diag']
        del g['diag', 'causes', 'diag']
        del g['diag', 'causes', 'pro']
        del g['pro', 'causes', 'diag']
        del g['diag', 'coexistswith', 'diag']
        del g['diag', 'coexistswith', 'pro']
        del g['pro', 'coexistswith', 'pro']
        del g['diag', 'complicates', 'diag']
        del g['pro', 'complicates', 'diag']
        del g['diag', 'diagnoses', 'diag']
        del g['pro', 'diagnoses', 'diag']
        del g['diag', 'isa', 'diag']
        del g['pro', 'isa', 'pro']
        del g['diag', 'manifestationof', 'diag']
        del g['pro', 'measures', 'diag']
        del g['pro', 'methodof', 'pro']
        del g['diag', 'precedes', 'diag']
        del g['pro', 'precedes', 'diag']
        del g['pro', 'precedes', 'pro']
        del g['diag', 'predisposes', 'diag']
        del g['pro', 'predisposes', 'diag']
        del g['pro', 'prevents', 'diag']
        del g['pro', 'treats', 'diag']
        return g

    def ckg_remove_ontology(self, g):
        del g['diag', 'icd9ontology', 'diag']
        del g['pro', 'icd9ontology', 'pro']
        return g

    def ckg_remove_demographics(self, g):
        del g['diag', 'belongsto', 'patient']
        del g['pro', 'belongsto', 'patient']
        del g['patient']
        return g

    def forward(self, g_c, g_m, y_m):

        # Initialized hidden states
        h_c_x = Variable(torch.zeros(1, self.args.dim)).to(self.device)
        if self.args.with_meds == 1:
            h_m_x = Variable(torch.zeros(1, self.args.dim)).to(self.device)
            h_cm_x = Variable(torch.zeros(1, self.args.dim)).to(self.device)

        for t in np.arange(len(g_c)):

            # Clinical Graph Representation Learning
            if self.args.with_demographics == 1:
                patient_cx = torch.cat([self.f_embed_dict['patient_gender'](g_c[t]['patient'].x[:2].argmax()).unsqueeze(0),
                                        self.f_embed_dict['patient_ethnicity'](g_c[t]['patient'].x[2:9].argmax()).unsqueeze(0),
                                        self.f_embed_dict['patient_age'](g_c[t]['patient'].x[9:].argmax()).unsqueeze(0)], dim=0)
                g_c[t]['patient'].x = patient_cx.sum(0, keepdim=True)
            g_c[t]['diag'].x = self.f_embed_dict['diag'](g_c[t]['diag'].x.long())
            g_c[t]['pro'].x = self.f_embed_dict['pro'](g_c[t]['pro'].x.long())
            if self.args.with_ontology == 0:
                g_c[t] = self.ckg_remove_ontology(g_c[t])
            if self.args.with_semantics == 0:
                g_c[t] = self.ckg_remove_semantics(g_c[t])
            if self.args.with_demographics == 0:
                g_c[t] = self.ckg_remove_demographics(g_c[t])
            g_c_t = g_c[t].to_homogeneous()
            c_x, c_edge_index, c_edge_type = g_c_t.x, g_c_t.edge_index, g_c_t.edge_type
            # Graph Augmentation
            if self.args.with_augment and self.training:
                c_x, c_edge_index, c_edge_type = augment_graph(c_x, c_edge_index, c_edge_type, n_edge_types=self.num_diagpro_relations)
            c_x = self.dropout(self.relu(self.c_conv1(c_x, c_edge_index, c_edge_type)))
            c_x = self.relu(self.c_conv2(c_x, c_edge_index, c_edge_type))
            c_x = self.readout(c_x)

            # Temporal Learning For Clinical
            h_c_x = self.c_rnn(c_x, h_c_x)

            # Medicine Graph  Representation Learning for past medication
            if self.args.with_meds == 1 and t < len(g_c)-1:
                if self.args.with_demographics == 1:
                    patient_mx = torch.cat([self.f_embed_dict['patient_gender'](g_m[t]['patient'].x[:2].argmax()).unsqueeze(0),
                                            self.f_embed_dict['patient_ethnicity'](g_m[t]['patient'].x[2:9].argmax()).unsqueeze(0),
                                            self.f_embed_dict['patient_age'](g_m[t]['patient'].x[9:].argmax()).unsqueeze(0)], dim=0)
                    g_m[t]['patient'].x = patient_mx.sum(0, keepdim=True)
                g_m[t]['med'].x = self.f_embed_dict['med'](g_m[t]['med'].x.long())
                if self.args.with_ontology == 0:
                    g_m[t] = self.mkg_remove_ontology(g_m[t])
                if self.args.with_ddi == 0:
                    g_m[t] = self.mkg_remove_ddi(g_m[t])
                if self.args.with_demographics == 0:
                    g_m[t] = self.mkg_remove_demographics(g_m[t])
                g_m_t = g_m[t].to_homogeneous()
                m_x, m_edge_index, m_edge_type = g_m_t.x, g_m_t.edge_index, g_m_t.edge_type
                # Graph Augmentation
                if self.args.with_augment and self.training:
                    m_x, m_edge_index, m_edge_type = augment_graph(m_x, m_edge_index, m_edge_type, n_edge_types=self.num_med_relations)
                m_x = self.dropout(self.relu(self.m_conv1(m_x, m_edge_index, m_edge_type)))
                m_x = self.relu(self.m_conv2(m_x, m_edge_index, m_edge_type))
                m_x = self.readout(m_x)

                # Temporal Learning for Medicine
                h_m_x = self.m_rnn(m_x, h_m_x)

                # Hierarchical RNNs with NCF-based Fusion
                if self.args.with_fusion == 1:
                    c_m_x = self.neucf(h_c_x, h_m_x)
                else:
                    c_m_x = torch.concat([h_c_x, h_m_x], dim=1)
                h_cm_x = self.cm_rnn(c_m_x, h_cm_x)

        # Get the recommended medicines as the output
        if self.args.with_meds == 0:
            if self.args.with_apm == 0:
                h_all = h_c_x
            else:
                h_all = self.norm(h_c_x + self.dropout(self.mha_triad(h_c_x, h_c_x, h_c_x, need_weights=False)[0]))
        else:
            if self.args.with_apm == 0:
                h_all = self.norm(c_x + self.dropout(self.merge(torch.concat([h_c_x, h_cm_x], dim=1))))
            else:
                h_all = self.norm(c_x + self.dropout(self.mha_triad(h_c_x, h_cm_x, h_cm_x, need_weights=False)[0]))
        y_logit = self.f_out(h_all)
        y_prob = F.sigmoid(y_logit)

        return y_logit, y_prob
