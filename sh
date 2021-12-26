  File "/home/hong/CL/pre-training/aa_train.py", line 343, in <module>
    aug2=b[0], aug_ratio2=b[1], suffix=None)
  File "/home/hong/CL/pre-training/train_eval.py", line 109, in controller_train
    model, optimizer, dataset, device, batch_size, aug1, aug_ratio1, aug2, aug_ratio2)
  File "/home/hong/CL/pre-training/train_eval.py", line 281, in train
    out1 = model.forward_cl(data1)
  File "/home/hong/CL/pre-training/res_gcn.py", line 173, in forward_cl
    return self.forward_BNConvReLU_cl(x, edge_index, batch, xg)
  File "/home/hong/CL/pre-training/res_gcn.py", line 183, in forward_BNConvReLU_cl
    x = self.global_pool(x * gate, batch)
  File "/home/hong/anaconda3/envs/hong2/lib/python3.7/site-packages/torch_geometric/nn/glob/glob.py", line 23, in global_add_pool
    size = batch[-1].item() + 1 if size is None else size
IndexError: index -1 is out of bounds for dimension 0 with size 0
(hong2) hong@hong-desktop:~/CL$ 




python main_cl.py --dataset NCI1 --aug1 dropN --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 1  61.53 5.80
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1  63.16 4.50
python main_cl.py --dataset NCI1 --aug1 permE --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 1   59.29 2.9
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 60.97 5.77 
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 




CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --dataset NCI1 --aug1 dropN --aug2 permE --lr 0.001 --suffix 0

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 permE --semi_split 10 --model_epoch 100 --suffix 5
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 2

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2

python main_cl.py --dataset NCI1 --aug1 permE --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 1
python main_cl.py --dataset NCI1 --aug1 permE --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 2

python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2

python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 
----------------------------------------------------------------------------------------------------------

c



python main.py --dataset NCI1 --aug1 dropN --aug2 permE --suffix 5 --alpha 0.01
python main.py --dataset NCI1 --aug1 dropN --aug2 permE --suffix 1 --alpha 1
python main.py --dataset NCI1 --aug1 dropN --aug2 permE --suffix 2 --alpha 1
   
python main.py --dataset NCI1 --aug1 dropN --aug2 maskN --suffix 0 --alpha 1
python main.py --dataset NCI1 --aug1 dropN --aug2 maskN --suffix 1 --alpha 1
python main.py --dataset NCI1 --aug1 dropN --aug2 maskN --suffix 2 --alpha 1

python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 1
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 1
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 1

python main.py --dataset NCI1 --aug1 permE --aug2 maskN --suffix 0 --alpha 1
python main.py --dataset NCI1 --aug1 permE --aug2 maskN --suffix 1 --alpha 1
python main.py --dataset NCI1 --aug1 permE --aug2 maskN --suffix 2 --alpha 1

python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 0 --alpha 1
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 1 --alpha 1
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 2 --alpha 1

python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 0 --alpha 1
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 1 --alpha 1
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 2 --alpha 1

-----------------------------------------------------------------------------------------------------------

python main.py --dataset PROTEINS --aug1 dropN --aug2 permE --suffix 0 --alpha 0.015
python main.py --dataset PROTEINS --aug1 dropN --aug2 permE --suffix 1 --alpha 0.015
python main.py --dataset PROTEINS --aug1 dropN --aug2 permE --suffix 2 --alpha 0.015
   
python main.py --dataset PROTEINS --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.015
python main.py --dataset PROTEINS --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.015
python main.py --dataset PROTEINS --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.015

python main.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.015
python main.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.015
python main.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.015

python main.py --dataset PROTEINS --aug1 permE --aug2 maskN --suffix 0 --alpha 0.015
python main.py --dataset PROTEINS --aug1 permE --aug2 maskN --suffix 1 --alpha 0.015
python main.py --dataset PROTEINS --aug1 permE --aug2 maskN --suffix 2 --alpha 0.015

python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.015
python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.015
python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.015

python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.015
python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.015
python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.015
---------------------------------------------------------------------------------------------------------------------------

python main.py --dataset NCI1 --aug1 dropN --aug2 permE --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 permE --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 permE --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 
   
python main.py --dataset NCI1 --aug1 dropN --aug2 maskN --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 maskN --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 maskN --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset NCI1 --aug1 permE --aug2 maskN --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 permE --aug2 maskN --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 permE --aug2 maskN --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2


python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.07 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.07 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.07 --aug_ratio1 0.2 --aug_ratio2 0.2


python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 0 --alpha 0.07 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 1 --alpha 0.07 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 2 --alpha 0.07 --aug_ratio1 0.2 --aug_ratio2 0.2
--------------------------------------------------------------------------------------------------------------------------

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100--model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100 --model_epoch 100


python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  dropN --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  dropN --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  dropN --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  dropN --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  dropN --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10 --model_epoch 100



python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10--model_epoch 100

python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  permE --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100

python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100
python main_cl.py --dataset NCI1 --aug1  maskN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.3 --aug_ratio2 0.3 --semi_split 10 --model_epoch 100



----------------------------------------
./go_NCI1.sh NCI1 0.001 100
./go_NCI1.sh NCI1 0.005 100
./go_NCI1.sh NCI1 0.008 100
./go_NCI1.sh NCI1 0.01 100
./go_NCI1.sh NCI1 0.03 100
./go_NCI1.sh NCI1 0.05 100
./go_NCI1.sh NCI1 0.1 100
./go_NCI1.sh NCI1 0.5 100
-------------------------
./go_NCI1.sh NCI1 0.01 
./go_NCI1.sh NCI1 0.03 
./go_NCI1.sh NCI1 0.05 
./go_NCI1.sh NCI1 0.1 
./go_NCI1.sh NCI1 0.5


./go_DD.sh DD 0.001 
./go_DD.sh DD 0.005 
./go_DD.sh DD 0.008 
./go_DD.sh DD 0.01 
./go_DD.sh DD 0.03 
./go_DD.sh DD 0.05 
./go_DD.sh DD 0.1 
./go_DD.sh DD 0.5 


--------------------------
./go_PROTEINS.sh PROTEINS 0.001 
./go_PROTEINS.sh PROTEINS 0.005 
./go_PROTEINS.sh PROTEINS 0.008 
./go_PROTEINS.sh PROTEINS 0.01 
./go_PROTEINS.sh PROTEINS 0.03 
./go_PROTEINS.sh PROTEINS 0.05 
./go_PROTEINS.sh PROTEINS 0.1 
./go_PROTEINS.sh PROTEINS 0.5
----------------------------

./go.sh NCI1 0.001 
./go.sh NCI1 0.005 
./go.sh NCI1 0.008 
./go.sh NCI1 0.01 
./go.sh NCI1 0.03 
./go.sh NCI1 0.05 
./go.sh NCI1 0.1 
./go.sh NCI1 0.5 
./go.sh NCI1 0.001 
./go.sh NCI1 0.005 
./go.sh NCI1 0.008 
./go.sh NCI1 0.01 
./go.sh NCI1 0.03 
./go.sh NCI1 0.05 
./go.sh NCI1 0.1 
./go.sh NCI1 0.5 


./go.sh PROTEINS 0.001 
./go.sh PROTEINS 0.005 
./go.sh PROTEINS 0.008 
./go.sh PROTEINS 0.01 
./go.sh PROTEINS 0.03 
./go.sh PROTEINS 0.05 
./go.sh PROTEINS 0.1 
./go.sh PROTEINS 0.5 
./go.sh PROTEINS 0.001 
./go.sh PROTEINS 0.005 
./go.sh PROTEINS 0.008 
./go.sh PROTEINS 0.01 
./go.sh PROTEINS 0.03 
./go.sh PROTEINS 0.05 
./go.sh PROTEINS 0.1 
./go.sh PROTEINS 0.5 


./go.sh DD 0.001 
./go.sh DD 0.005 
./go.sh DD 0.008 
./go.sh DD 0.01 
./go.sh DD 0.03 
./go.sh DD 0.05 
./go.sh DD 0.1 
./go.sh DD 0.5 

./go.sh DD 0.001 150
./go.sh DD 0.005 150
./go.sh DD 0.008 150
./go.sh DD 0.01 150
./go.sh DD 0.03 150
./go.sh DD 0.05 150
./go.sh DD 0.1 150
./go.sh DD 0.5 150
-------------------------


./go.sh PROTEINS 0.001 100
./go.sh PROTEINS 0.005 100
./go.sh PROTEINS 0.008 100
./go.sh PROTEINS 0.01 100
./go.sh PROTEINS 0.03 100
./go.sh PROTEINS 0.05 100
./go.sh PROTEINS 0.1 100
./go.sh PROTEINS 0.5 100
./go.sh PROTEINS 0.001 150
./go.sh PROTEINS 0.005 150
./go.sh PROTEINS 0.008 150
./go.sh PROTEINS 0.01 150
./go.sh PROTEINS 0.03 150
./go.sh PROTEINS 0.05 150
./go.sh PROTEINS 0.1 150
./go.sh PROTEINS 0.5 150


./go.sh DD 0.001 100
./go.sh DD 0.005 100
./go.sh DD 0.008 100
./go.sh DD 0.01 100
./go.sh DD 0.03 100
./go.sh DD 0.05 100
./go.sh DD 0.1 100
./go.sh DD 0.5 100
./go.sh DD 0.001 150
./go.sh DD 0.005 150
./go.sh DD 0.008 150
./go.sh DD 0.01 150
./go.sh DD 0.03 150
./go.sh DD 0.05 150
./go.sh DD 0.1 150
./go.sh DD 0.5 150


./go_COLLAB.sh COLLAB 0.001 
./go_COLLAB.sh COLLAB 0.005 
./go_COLLAB.sh COLLAB 0.008 
./go_COLLAB.sh COLLAB 0.01 
./go_COLLAB.sh COLLAB 0.03 
./go_COLLAB.sh COLLAB 0.05 
./go_COLLAB.sh COLLAB 0.1 
./go_COLLAB.sh COLLAB 0.5 

./go.sh COLLAB 0.001 150
./go.sh COLLAB 0.005 150
./go.sh COLLAB 0.008 150
./go.sh COLLAB 0.01 150
./go.sh COLLAB 0.03 150
./go.sh COLLAB 0.05 150
./go.sh COLLAB 0.1 150
./go.sh COLLAB 0.5 150
./go.sh COLLAB 0.008 150
---------------------------------------------
./go.sh REDDIT-BINARY 0.001 100
./go.sh REDDIT-BINARY 0.005 100
./go.sh REDDIT-BINARY 0.008 100
./go.sh REDDIT-BINARY 0.01 100
./go.sh REDDIT-BINARY 0.03 100
./go.sh REDDIT-BINARY 0.05 100
./go.sh REDDIT-BINARY 0.1 100
./go.sh REDDIT-BINARY 0.5 100
./go.sh REDDIT-BINARY 0.001 150
./go.sh REDDIT-BINARY 0.005 150
./go.sh REDDIT-BINARY 0.008 150
./go.sh REDDIT-BINARY 0.01 150
./go.sh REDDIT-BINARY 0.03 150
./go.sh REDDIT-BINARY 0.05 150
./go.sh REDDIT-BINARY 0.1 150
./go.sh REDDIT-BINARY 0.5 150
./go.sh REDDIT-BINARY 0.008 150


./go.sh REDDIT-MULTI-5K 0.001 100
./go.sh REDDIT-MULTI-5K 0.005 100
./go.sh REDDIT-MULTI-5K 0.008 100
./go.sh REDDIT-MULTI-5K 0.01 100
./go.sh REDDIT-MULTI-5K 0.03 100
./go.sh REDDIT-MULTI-5K 0.05 100
./go.sh REDDIT-MULTI-5K 0.1 100
./go.sh REDDIT-MULTI-5K 0.5 100
./go.sh REDDIT-MULTI-5K 0.001 150
./go.sh REDDIT-MULTI-5K 0.005 150
./go.sh REDDIT-MULTI-5K 0.008 150
./go.sh REDDIT-MULTI-5K 0.01 150
./go.sh REDDIT-MULTI-5K 0.03 150
./go.sh REDDIT-MULTI-5K 0.05 150
./go.sh REDDIT-MULTI-5K 0.1 150
./go.sh REDDIT-MULTI-5K 0.5 150
./go.sh REDDIT-MULTI-5K 0.008 150






-----------------------------------------------------

python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.02 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.02 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.02 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.02 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.02 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2


python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2



python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.08 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.08 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.08 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.08 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.08 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100

python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 100

python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.4 --aug_ratio1 0.2 --aug_ratio2 0.2








python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.001 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.001 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.001 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.001 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.001 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2

python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.2 --aug_ratio1 0.2 --aug_ratio2 0.2

python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python asa2.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2


python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.1 --aug_ratio1 0.2 --aug_ratio2 0.2


python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.3 --aug_ratio1 0.2 --aug_ratio2 0.2


python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.05 --aug_ratio1 0.2 --aug_ratio2 0.2





python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.2
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3

python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3

python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3
python main.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.3 --aug_ratio2 0.3


python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.0015 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.0015 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.0015 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.0015 --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset NCI1 --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.0015 --aug_ratio1 0.2 --aug_ratio2 0.2
-----------------------------------------------------------------------------------------------------


python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 3 --alpha 0
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 4 --alpha 0


python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 3 --alpha 0
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 4 --alpha 0


python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 3 --alpha 0
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 4 --alpha 0


python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 3 --alpha 0
python main_cl.py --dataset NCI1 --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 4 --alpha 0


python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 3 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 4 --alpha 0.008


python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 3 --alpha 0.008
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 4 --alpha 0.008


 ------------------------------------------------------------------------------------------------------------------------

python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2

python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.008  --aug_ratio1 0.2 --aug_ratio2 0.2 

python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main_cl.py --dataset github_stargazers --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.008 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2

python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 

python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 
python main.py --dataset REDDIT-MULTI-5K --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.2 --aug_ratio2 0.2 


python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10

python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10

python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10

python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10

python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10

python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0 --aug_ratio1 0.2 --aug_ratio2 0.2 --semi_split 10







python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0

python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 permE --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 permE --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 permE --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0

python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 maskN --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0

python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 0 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 1 --alpha 0
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --semi_split 100 --model_epoch 100 --suffix 2 --alpha 0


python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 maskN --semi_split 10 --model_epoch 100 --suffix 0 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 maskN --semi_split 10 --model_epoch 100 --suffix 1 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 maskN --semi_split 10 --model_epoch 100 --suffix 2 --alpha 1

python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 permE --semi_split 10 --model_epoch 100 --suffix 0 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 permE --semi_split 10 --model_epoch 100 --suffix 1 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 permE --semi_split 10 --model_epoch 100 --suffix 2 --alpha 1

python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 0 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 1 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 dropN --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 2 --alpha 1

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 maskN --semi_split 10 --model_epoch 100 --suffix 0 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 maskN --semi_split 10 --model_epoch 100 --suffix 1 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 maskN --semi_split 10 --model_epoch 100 --suffix 2 --alpha 1

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 0 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 1 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 2 --alpha 1

python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 0 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 1 --alpha 1
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --semi_split 10 --model_epoch 100 --suffix 2 --alpha 1



------------------------------------------------------------------------------------------------------------------------

python main.py --dataset DD --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main.py --dataset DD --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset DD --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset DD --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01



python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01
python main.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01

python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.01
python main.py --dataset DD --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.01



python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.01

python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01



--------------------------------------------------------------------------------------


python main.py --dataset COLLAB --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main.py --dataset COLLAB --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main.py --dataset COLLAB --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main.py --dataset COLLAB --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset COLLAB --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset COLLAB --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset COLLAB --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset COLLAB --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset COLLAB --aug1 permE --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset COLLAB --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset COLLAB --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset COLLAB --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01


python main.py --dataset github_stargazers --aug1 dropN --aug2 permE --suffix 0 --alpha 0
python main.py --dataset github_stargazers  --aug1 dropN --aug2 permE --suffix 1 --alpha 0
python main.py --dataset github_stargazers  --aug1 dropN --aug2 permE --suffix 2 --alpha 0

python main.py --dataset github_stargazers  --aug1 dropN --aug2 maskN --suffix 0 --alpha 0
python main.py --dataset github_stargazers  --aug1 dropN --aug2 maskN --suffix 1 --alpha 0
python main.py --dataset github_stargazers  --aug1 dropN --aug2 maskN --suffix 2 --alpha 0

python main.py --dataset github_stargazers  --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0
python main.py --dataset github_stargazers  --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0
python main.py --dataset github_stargazers  --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0

python main.py --dataset github_stargazers  --aug1 permE --aug2 maskN --suffix 0 --alpha 0
python main.py --dataset github_stargazers  --aug1 permE --aug2 maskN --suffix 1 --alpha 0
python main.py --dataset github_stargazers  --aug1 permE --aug2 maskN --suffix 2 --alpha 0

python main.py --dataset github_stargazers  --aug1 permE --aug2 subgraph --suffix 0 --alpha 0
python main.py --dataset github_stargazers  --aug1 permE --aug2 subgraph --suffix 1 --alpha 0
python main.py --dataset github_stargazers  --aug1 permE --aug2 subgraph --suffix 2 --alpha 0

python main.py --dataset github_stargazers  --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0
python main.py --dataset github_stargazers  --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0
python main.py --dataset github_stargazers  --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0


python main.py --dataset REDDIT-BINARY --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-BINARY  --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 permE --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-BINARY  --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-BINARY  --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-BINARY  --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01




python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 maskN --suffix 2--alpha 0.01

python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-BINARY --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01

----------------------------------------------------------------------------------------------------

python main_cl.py --dataset DD --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main_cl.py --dataset DD --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01 
python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset DD --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 maskN --suffix 2--alpha 0.01

python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset DD --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset DD --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset DD --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01


python main_cl.py --dataset COLLAB --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main_cl.py --dataset COLLAB --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset COLLAB --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 permE --aug2 maskN --suffix 2 --alpha 0.01

python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01 
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 permE --aug2 subgraph --suffix 2--alpha 0.01

python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset COLLAB --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01


python main_cl.py --dataset GITHUB --aug1 dropN --aug2 permE --suffix 0 
python main_cl.py --dataset GITHUB --aug1 dropN --aug2 permE --suffix 1 
python main_cl.py --dataset GITHUB --aug1 dropN --aug2 permE --suffix 2 

python main_cl.py --dataset GITHUB --aug1 dropN --aug2 maskN --suffix 0 
python main_cl.py --dataset GITHUB --aug1 dropN --aug2 maskN --suffix 1 
python main_cl.py --dataset GITHUB --aug1 dropN --aug2 maskN --suffix 2 

python main_cl.py --dataset GITHUB --aug1 dropN --aug2 subgraph --suffix 0 
python main_cl.py --dataset GITHUB --aug1 dropN --aug2 subgraph --suffix 1 
python main_cl.py --dataset GITHUB --aug1 dropN --aug2 subgraph --suffix 2

python main_cl.py --dataset GITHUB --aug1 permE --aug2 maskN --suffix 0 
python main_cl.py --dataset GITHUB --aug1 permE --aug2 maskN --suffix 1 
python main_cl.py --dataset GITHUB --aug1 permE --aug2 maskN --suffix 2 

python main_cl.py --dataset GITHUB --aug1 permE --aug2 subgraph --suffix 0 
python main_cl.py --dataset GITHUB --aug1 permE --aug2 subgraph --suffix 1 
python main_cl.py --dataset GITHUB --aug1 permE --aug2 subgraph --suffix 2 

python main_cl.py --dataset GITHUB --aug1 maskN --aug2 subgraph --suffix 0 
python main_cl.py --dataset GITHUB --aug1 maskN --aug2 subgraph --suffix 1 
python main_cl.py --dataset GITHUB --aug1 maskN --aug2 subgraph --suffix 2 









python main.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-MULTI-5K  --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01

python main.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 maskN --suffix 2 --alpha 0.01


python main_cl.py --dataset REDDIT-MULTI-5K --aug1 dropN --aug2 permE --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 permE --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 permE --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 maskN --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 maskN --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 maskN --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 maskN --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01

python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset REDDIT-MULTI-5K  --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01




















python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 0 --alpha 0.01
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 1 --alpha 0.01
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 2 --alpha 0.01
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 3 --alpha 0.01
python main_cl.py --dataset NCI1 --aug1 dropN --aug2 subgraph --suffix 4 --alpha 0.01


python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4

python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4
python main_cl.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.01 --aug_ratio1 0.4 --aug_ratio2 0.4


python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 0 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 1 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 2 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 3 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 maskN --aug2 subgraph --suffix 4 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4

python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 0 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 1 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 2 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 3 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
python main.py --dataset PROTEINS --aug1 permE --aug2 subgraph --suffix 4 --alpha 0.02 --aug_ratio1 0.4 --aug_ratio2 0.4
















