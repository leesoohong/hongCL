#!/bin/bash -ex

for epoch in 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250  
do
  python main_cl_his.py --dataset "NCI1" --lr 0.001 --aug1 "dropN" --aug2 "subgraph" --alpha 0.001 --epochs $epoch --suffix 0
done

