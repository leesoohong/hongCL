import sys
import torch.nn as nn
import time
import random
import torch
import torch.nn.functional as F
from torch import normal, tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from torch_geometric.data.dataloader import DenseDataLoader
import matplotlib.pyplot as plt
from utils import print_weights
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def single_train_test(train_dataset,
                      test_dataset,
                      model_func,
                      epochs,
                      batch_size,
                      lr,
                      lr_decay_factor,
                      lr_decay_step_size,
                      weight_decay,
                      epoch_select,
                      with_eval_mode=True):
    # assert epoch_select in ['test_last', 'test_max'], epoch_select

    model = model_func(train_dataset).to(device)
    print_weights(model)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    train_accs, test_accs = [], []
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_loss, train_acc = train(
            model, optimizer, train_loader, device)
        train_accs.append(train_acc)
        test_accs.append(eval_acc(model, test_loader, device, with_eval_mode))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        print('Epoch: {:03d}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(
            epoch, train_accs[-1], test_accs[-1]))
        sys.stdout.flush()

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

    t_end = time.perf_counter()
    duration = t_end - t_start

    if epoch_select == 'test_max':
        train_acc = max(train_accs)
        test_acc = max(test_accs)
    else:
        train_acc = train_accs[-1]
        test_acc = test_accs[-1]

    return train_acc, test_acc, duration
from copy import deepcopy


def cross_validation_with_val_set(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  alpha,
                                  with_eval_mode=True,
                                  logger=None,
                                  dataset_name=None,
                                  aug1=None, aug_ratio1=None,
                                  aug2=None, aug_ratio2=None, suffix=None):
    # assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select))):
        
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        for batch in train_loader:
            print(batch)
        

        dataset.aug = "none"
        model = model_func(dataset).to(device)
        if fold == 0:
            print_weights(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        for epoch in range(1, epochs + 1):


            
            train_loss= train(
                model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2,alpha)
            print(train_loss)
        

            # with open('/home/hong/CL/pre-training/logs/' + dataset_name + 'autoaugment2_cl_log', 'a+') as f:
            #         f.write(str(epoch) + ' ' + str(train_loss))
            #         f.write('\n')

            
        torch.save(model.state_dict(), '/home/hong/CL/pre-training/models5/' + dataset_name + '_' + aug1 + '_' + str(aug_ratio1) + '_'+ aug2 + '_' + str(aug_ratio2) + '_' + str(lr) + '_' + str(suffix) +"_"+str(alpha)+ "_"+str(epoch)+'.pt')
        acc= eval_acc(model,train_loader, device, with_eval_mode)
        print(acc)
        print("finish run")
        break



def cross_validation_with_val_set_his(dataset,
                                  model_func,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  epoch_select,
                                  alpha,
                                  with_eval_mode=True,
                                  logger=None,
                                  dataset_name=None,
                                  aug1=None, aug_ratio1=None,
                                  aug2=None, aug_ratio2=None, suffix=None):
    # assert epoch_select in ['val_max', 'test_max'], epoch_select

    val_losses, train_accs, test_accs, durations = [], [], [], []
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds, epoch_select))):
        
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        for batch in train_loader:
            print(batch)
        

        dataset.aug = "none"
        model = model_func(dataset).to(device)
        if fold == 0:
            print_weights(model)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()
        sum_loss_vals=[]
        normal_loss_vals=[]
        consis_loss_vals=[]
   
        for epoch in range(1, epochs + 1):


            sum_loss=[]
            consis_loss=[]
            normal_loss=[]
            train_loss,sum_loss,normal_loss,consis_loss = train_history(
                model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2, alpha,sum_loss,normal_loss,consis_loss)
            print(train_loss)
            sum_loss_vals.append(sum(sum_loss)/len(sum_loss))
            normal_loss_vals.append(sum(normal_loss)/len(normal_loss))
            consis_loss_vals.append(sum(consis_loss)/len(consis_loss))
            # if epoch /100==0:
            #     torch.save(model.state_dict(), '/home/hong/CL/pre-training/model_his/' + dataset_name + '_' + aug1 + '_' + str(aug_ratio1) + '_'+ aug2 + '_' + str(aug_ratio2) + '_' + str(lr) + '_' + str(suffix) +"_"+str(alpha)+ "_"+str(epoch)+'.pt')

            # with open('/home/hong/CL/pre-training/logs/' + dataset_name + 'autoaugment2_cl_log', 'a+') as f:
            #         f.write(str(epoch) + ' ' + str(train_loss))
            #         f.write('\n')

        # torch.save(model.state_dict(), '/home/hong/CL/pre-training/model_his/' + dataset_name + '_' + aug1 + '_' + str(aug_ratio1) + '_'+ aug2 + '_' + str(aug_ratio2) + '_' + str(lr) + '_' + str(suffix) +"_"+str(alpha)+ "_"+str(epoch)+'.pt')
        acc= eval_acc(model,train_loader, device, with_eval_mode)
        print(acc)
        print("finish run")
        plt.subplot(221)
        plt.plot(np.linspace(1,epochs,epochs).astype(int),sum_loss_vals,label="total_loss",color='r')
        plt.plot(np.linspace(1,epochs,epochs).astype(int),normal_loss_vals,label="normal_loss",color='g')
        plt.savefig('pic_'+str(epoch)+'_1png')
        plt.subplot(222)
        plt.plot(np.linspace(1,epochs,epochs).astype(int),consis_loss_vals,label="consis_loss",color='b')
        plt.savefig('pic_'+str(epoch)+'_2png')

        plt.show()
        
        break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    """
    duration = tensor(durations)
    train_acc, test_acc = tensor(train_accs), tensor(test_accs)
    val_loss = tensor(val_losses)
    train_acc = train_acc.view(folds, epochs)
    test_acc = test_acc.view(folds, epochs)
    val_loss = val_loss.view(folds, epochs)
    if epoch_select == 'test_max':  # take epoch that yields best test results.
        _, selected_epoch = test_acc.mean(dim=0).max(dim=0)
        selected_epoch = selected_epoch.repeat(folds)
    else:  # take epoch that yields min val loss for each fold individually.
        _, selected_epoch = val_loss.min(dim=1)
    test_acc = test_acc[torch.arange(folds, dtype=torch.long), selected_epoch]
    train_acc_mean = train_acc[:, -1].mean().item()
    test_acc_mean = test_acc.mean().item()
    test_acc_std = test_acc.std().item()
    duration_mean = duration.mean().item()
    print('Train Acc: {:.4f}, Test Acc: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(train_acc_mean, test_acc_mean, test_acc_std, duration_mean))
    sys.stdout.flush()
    return train_acc_mean, test_acc_mean, test_acc_std, duration_mean
    """


def k_fold(dataset, folds, epoch_select):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx))

    if epoch_select == 'test_max':
        val_indices = [test_indices[i] for i in range(folds)]
    else:
        val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i].long()] = 0
        train_mask[val_indices[i].long()] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


# def train2(model, optimizer, dataset, device, batch_size):
    aug=[["maskN",0.1,"dropN",0.2],
                ["dropnN",0.2,"subgraph",0.4],
                ["subgraph",0.4,"maskN",0.1],
                ["maskN",0.1,"permE",0.2],
                ["maskN",0,"maskN",0.4]]

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    aug_order=[]

    
    for data1 in dataset1:
        aug_idx = random.randint(0, len(aug) - 1)
        aug_num=[0,0,0,0,0]
        data1.aug, data1.aug_ratio = aug[aug_idx][0], aug[aug_idx][1]
        aug_num[aug_idx]+=1
        aug_order.append(aug_idx)
    for data2 in dataset2:
        i=0
        data2.aug, data2.aug_ratio = aug[aug_num[i]][2], aug[aug_num[i]][3]
        i+=1


    
    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)
        

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2 in zip(loader1, loader2):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward_cl(data1)
        out2 = model.forward_cl(data2)
        loss = model.loss_cl(out1, out2)
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset), aug_num
def train3(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2,alpha):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset3=deepcopy(dataset)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)
    loader3 =DataLoader(dataset3, batch_size, shuffle=False)
    

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2 ,data3 in zip(loader1, loader2,loader3):

        data1 = data1.to(device)
        data2 = data2.to(device)
        data3=  data3.to(device)
        out1 = model.forward_cl(data1)
        out2 = model.forward_cl(data2)
        out3 = model.forward_cl(data3)

        x1_abs = out1.norm(dim=1)
        x2_abs =out2.norm(dim=1)
        x3_abs =out3.norm(dim=1)


    
        sim_matrix1= torch.einsum('ik,jk->ij', out3, out1) / torch.einsum('i,j->ij', x3_abs, x1_abs)
        sim_matrix2= torch.einsum('ik,jk->ij', out3, out2) / torch.einsum('i,j->ij', x3_abs, x2_abs)
        print(sim_matrix1)
        print(sim_matrix2)

        m = nn.Softmax(dim=1)
        sim_matrix1=m(sim_matrix1/0.5)
        sim_matrix2=m(sim_matrix2/0.5)
       
        loss_consis=1/2*torch.nn.functional.kl_div(sim_matrix1,sim_matrix2,reduction='sum')+1/2*torch.nn.functional.kl_div(sim_matrix2,sim_matrix1,reduction='sum')
        print(loss_consis.shape)
        print(loss_consis)
        loss_consis=loss_consis*alpha
        # loss_consis= - torch.log(loss_consis)
        loss = model.loss_cl(out1, out2)+loss_consis
        # loss = model.loss_cl(out1, out2)

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset) 


def train_kl(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2,alpha):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset3=deepcopy(dataset)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)
    loader3 =DataLoader(dataset3, batch_size, shuffle=False)
    

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2 ,data3 in zip(loader1, loader2,loader3):

        data1 = data1.to(device)
        data2 = data2.to(device)
        data3=  data3.to(device)
        out1 = model.forward_cl(data1)
        out2 = model.forward_cl(data2)
        out3 = model.forward_cl(data3)

        x1_abs = out1.norm(dim=1)
        x2_abs =out2.norm(dim=1)
        x3_abs =out3.norm(dim=1)


    
        sim_matrix1= torch.einsum('ik,jk->ij', out3, out1) / torch.einsum('i,j->ij', x3_abs, x1_abs)
        sim_matrix2= torch.einsum('ik,jk->ij', out3, out2) / torch.einsum('i,j->ij', x3_abs, x2_abs)
        F.log_softmax(sim_matrix1,dim=1)
        F.log_softmax(sim_matrix2,dim=1)
        loss_consis=1/2*torch.nn.functional.kl_div(sim_matrix1,sim_matrix2,reduction='sum')+1/2*torch.nn.functional.kl_div(sim_matrix2,sim_matrix1,reduction='sum')
        loss_consis=loss_consis*alpha
        # loss_consis= - torch.log(loss_consis)
        loss = model.loss_cl(out1, out2)+loss_consis
        # loss = model.loss_cl(out1, out2)

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset) 

def train(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2,alpha):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset3=deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)
    loader3 =DataLoader(dataset3, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2,data3 in zip(loader1, loader2,loader3):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3=data3.to(device)
        out1 = model.forward_cl(data1)
        out2 = model.forward_cl(data2)
        out3 = model.forward_cl(data3)


        
  
        # print(out1.shape)
        # print(out2.shape)
        
        
        
        x1_abs = out1.norm(dim=1)
        x2_abs =out2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', out1, out2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_t=torch.transpose(sim_matrix,0,1)
        m = nn.Softmax(dim=1)
        sim_matrix=m(sim_matrix/0.5)
        sim_matrix_t=m(sim_matrix_t/0.5)
      
    
        # loss_consis=1/2*torch.nn.functional.kl_div(sim_matrix.log(),sim_matrix_t,reduction='sum')
        loss_consis=1/2*torch.nn.functional.kl_div(sim_matrix.log(),sim_matrix_t,reduction='sum')+1/2*torch.nn.functional.kl_div(sim_matrix_t.log(),sim_matrix,reduction='sum')
        loss_consis=loss_consis*alpha
        # loss_consis= - torch.log(loss_consis)
        loss = model.loss_cl(out1, out2)+loss_consis
        # loss = model.loss_cl(out1, out2)

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset) 

def train_history(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2,alpha, sum_loss, normal_loss,consis_loss):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset3=deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)
    loader3 =DataLoader(dataset3, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    
    for data1, data2,data3 in zip(loader1, loader2,loader3):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3=data3.to(device)
        out1 = model.forward_cl(data1)
        out2 = model.forward_cl(data2)
        out3 = model.forward_cl(data3)


        x1_abs = out1.norm(dim=1)
        x2_abs =out2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', out1, out2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_t=torch.transpose(sim_matrix,0,1)
        m = nn.Softmax(dim=1)
        sim_matrix=m(sim_matrix/0.5)
        sim_matrix_t=m(sim_matrix_t/0.5)
      
    
        # loss_consis=1/2*torch.nn.functional.kl_div(sim_matrix.log(),sim_matrix_t,reduction='sum')
        loss_consis=1/2*torch.nn.functional.kl_div(sim_matrix.log(),sim_matrix_t,reduction='sum')+1/2*torch.nn.functional.kl_div(sim_matrix_t.log(),sim_matrix,reduction='sum')
        loss_consis=loss_consis*alpha
        # loss_consis= - torch.log(loss_consis)
        n_loss=model.loss_cl(out1, out2)
        loss = model.loss_cl(out1, out2)+loss_consis   
    
        # loss = model.loss_cl(out1, out2)   
        sum_loss.append(loss.item())
        normal_loss.append(n_loss.item())
        consis_loss.append(loss_consis.item())
        loss.backward()
        

        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset), sum_loss,normal_loss,consis_loss

def train_other(model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2,alpha):
    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset3=deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = aug1, aug_ratio1
    dataset2 = deepcopy(dataset1)
    dataset2.aug, dataset2.aug_ratio = aug2, aug_ratio2

    loader1 = DataLoader(dataset1, batch_size, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size, shuffle=False)
    loader3 =DataLoader(dataset3, batch_size, shuffle=False)

    model.train()

    total_loss = 0
    correct = 0
    for data1, data2,data3 in zip(loader1, loader2,loader3):
        # print(data1, data2)
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3=data3.to(device)
        out1 = model.forward_cl(data1)
        out2 = model.forward_cl(data2)
        out3 = model.forward_cl(data3)


        
  
        # print(out1.shape)
        # print(out2.shape)
        
        
        
        x1_abs = out1.norm(dim=1)
        x2_abs =out3.norm(dim=1)


        sim_matrix = torch.einsum('ik,jk->ij', out1, out3) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_t=torch.transpose(sim_matrix,0,1)
        m = nn.Softmax(dim=1)
        sim_matrix=m(sim_matrix/0.5)
        sim_matrix_t=m(sim_matrix_t/0.5)
        loss_consis1=1/2*torch.nn.functional.kl_div(sim_matrix.log(),sim_matrix_t,reduction='sum')+1/2*torch.nn.functional.kl_div(sim_matrix_t.log(),sim_matrix,reduction='sum')
        loss_consis1=loss_consis1*alpha

        x1_abs = out2.norm(dim=1)
        x2_abs =out3.norm(dim=1)


        sim_matrix = torch.einsum('ik,jk->ij', out2, out3) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix_t=torch.transpose(sim_matrix,0,1)
        m = nn.Softmax(dim=1)
        sim_matrix=m(sim_matrix/0.5)
        sim_matrix_t=m(sim_matrix_t/0.5)
        loss_consis2=1/2*torch.nn.functional.kl_div(sim_matrix.log(),sim_matrix_t,reduction='sum')+1/2*torch.nn.functional.kl_div(sim_matrix_t.log(),sim_matrix,reduction='sum')
        loss_consis2=loss_consis2*alpha


        # loss_consis= - torch.log(loss_consis)
        loss = model.loss_cl(out1, out2)+loss_consis1+loss_consis2
        # loss = model.loss_cl(out1, out2)

        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()
    return total_loss / len(loader1.dataset) 







def eval_acc(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)
    


def eval_loss(model, loader, device, with_eval_mode):
    if with_eval_mode:
        model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)




