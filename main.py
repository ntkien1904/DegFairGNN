import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from models import DFair_GCN, DFair_GAT, DFair_Sage #DFairGNN_2
from layers import Discriminator
import datasets
import datetime, time
import utils
import argparse
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
import scipy.stats as st
import math

def compute_CI(out_list, name=None, log_file=None):
    ci = 1.96 * st.sem(out_list) / math.sqrt(len(out_list))
    log = name + ' Mean: {:.4f} '.format(np.mean(out_list)) + \
            'Std: {:.4f}'.format(st.sem(out_list)) 
    print(log)


#Get parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='nba', help='dataset')

# model arguments
parser.add_argument('--d', type=int, default=1, help='degree evaluation')
parser.add_argument('--dim', type=int, default=32, help='hidden layer dimension')
parser.add_argument('--dim_d', type=int, default=32, help='degree mat dimension')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout percentage')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--omega', type=float, default=0.1, help='weight bias')
parser.add_argument('--base', type=int, default=1, help='1: GCN, 2: GAT, 3: Sage')
parser.add_argument('--k', type=float, default=1, help='ratio split head and tail group')

parser.add_argument('--w_b', type=float, default=1e-04, help='weight constraint')
parser.add_argument('--w_film', type=float, default=1e-04, help='weight FILM')
parser.add_argument('--w_f', type=float, default=1e-04, help='weight fair')
parser.add_argument('--decay', type=float, default=1e-04, help='weight decay')

# training arguments
parser.add_argument('--epochs', type=int, default=500, help='number of iteration')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
args = parser.parse_args()


cuda = torch.cuda.is_available()


class Controller(object):
    def __init__(self, model):
        self.model = model
        self.loss = F.nll_loss
        self.optim = optim.Adam(self.model.parameters(), 
                    lr=args.lr, weight_decay=args.decay)
       
     

    def train(self, data):
        print('Training ...')
        

        for i in range(args.epochs):
            self.model.train()
            self.optim.zero_grad()

            output, b, film = self.model(data.feat, data.adj, data.degree, data.idx_train, data.edge)               
            
            assert not torch.isnan(output).any()                         
            
            train_output = output[data.idx_train]
            mean = torch.mean(data.degree[data.idx_train].float())

            idx_low = torch.where(data.degree[data.idx_train] < mean)[0]
            idx_high = torch.where(data.degree[data.idx_train] >= mean)[0]

            low_embed = torch.mean(train_output[idx_low], dim=0)
            high_embed = torch.mean(train_output[idx_high], dim=0)


            sp_loss = F.mse_loss(low_embed, high_embed)

            low_eo_embed = torch.zeros(data.labels.max()+1).cuda()
            high_eo_embed = torch.zeros(data.labels.max()+1).cuda()

    
            train_label = data.labels[data.idx_train]
            for i in range(data.labels.max()+1):
                idx_lc = torch.where(train_label[idx_low] == i)[0]
                idx_hc = torch.where(train_label[idx_high] == i)[0]

                mean_l = torch.mean(train_output[idx_lc], dim=0)
                mean_h = torch.mean(train_output[idx_hc], dim=0)

                low_eo_embed[i] = mean_l[i]
                high_eo_embed[i] = mean_h[i]
            eo_loss = F.mse_loss(low_eo_embed, high_eo_embed) 

            L_cls = self.loss(output[data.idx_train], data.labels[data.idx_train])

            # no L3
            # b = 0
            loss = L_cls + (args.w_f * sp_loss) + (args.w_b * b) + (args.w_b * film) 
            loss.backward()
            self.optim.step()
        return


    def test(self, data):
        print('Testing ...')
        
        self.model.eval()
       
        # accuracy
        output, _, _ = self.model(data.feat, data.adj, data.degree, data.idx_test, data.edge)
        acc = utils.accuracy(output[data.idx_test], data.labels[data.idx_test])
        acc = acc.cpu()
        print('Accuracy={:.4f}'.format(acc))

        preds = output.max(1)[1].cpu().detach()
        macf = f1_score(data.labels[data.idx_test].cpu(), preds[data.idx_test], average='macro')

        out1 = utils.evaluate_fairness(preds, data, data.group1, embed=output[data.idx_test], name=args.dataset)
        out2 = utils.evaluate_fairness(preds, data, data.group2)

        return acc*100, macf*100, out1, out2


def main():
    print(str(args))

    num = 5
    np.random.seed(args.seed)
    seed = np.random.choice(100, num, replace=False)
    
    f_acc = list()
    f_macf = list()
    SP_1 = list()
    EO_1 = list()
    h_acc_1 = list()
    t_acc_1 = list()

    SP_2 = list()
    EO_2 = list()
    h_acc_2 = list()
    t_acc_2 = list()

    for i in range(num):
        print('Seed: ', seed[i])
        np.random.seed(seed[i])
        torch.manual_seed(seed[i])
        if cuda:
            torch.cuda.set_device(args.gpu)
            torch.cuda.manual_seed(seed[i])


        data = datasets.get_dataset(args.dataset, norm=False, degree=args.d)
        data.to_tensor()
        in_feat = data.feat.shape[1]
        in_class = data.labels.max().item() + 1


        if args.base == 1:
            data.adj = datasets.convert_sparse_tensor(sp.csc_matrix(data.adj))
            model = DFair_GCN(args, in_feat, in_class, data.max_degree)

        elif args.base == 2:
            data.edge = torch.FloatTensor(data.adj)
            data.edge = data.edge.nonzero(as_tuple=False).t()
            data.adj = datasets.convert_sparse_tensor(sp.csc_matrix(data.adj))
            model = DFair_GAT(args, in_feat, in_class, data.max_degree)

        elif args.base == 3:
            data.adj = datasets.convert_sparse_tensor(sp.csc_matrix(data.adj))
            model = DFair_Sage(args, in_feat, in_class, data.max_degree)

        else:
            ValueError('model invalid')

        if cuda:
            model.cuda()
            data.to_cuda()
            
        controller = Controller(model)
        controller.train(data)

        acc, macf, out1, out2 = controller.test(data)

        f_acc.append(acc)
        f_macf.append(macf)

        SP_1.append(out1[0])
        EO_1.append(out1[1])
        h_acc_1.append(out1[2])
        t_acc_1.append(out1[3])
     
        SP_2.append(out2[0])
        EO_2.append(out2[1])
        h_acc_2.append(out2[2])
        t_acc_2.append(out2[3])
      
        del model

    print('--------------------------------------------------------------')
    compute_CI(f_acc, name='Acc')

    print('Degree 1')
    compute_CI(h_acc_1, name='Head Acc')
    compute_CI(t_acc_1, name='Tail Acc')
    compute_CI(SP_1, name='SP')
    compute_CI(EO_1, name='EO')

    print('Degree 2')
    compute_CI(h_acc_2, name='Head Acc')
    compute_CI(t_acc_2, name='Tail Acc')
    compute_CI(SP_2, name='SP')
    compute_CI(EO_2, name='EO')

if __name__ == "__main__":
    main()
