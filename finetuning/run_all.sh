#!/bin/bash -ex

for suffix in 0 1 2 3 4 
do
  python main_cl.py --dataset $1 --lr 0.001 --aug1 "dropN" --aug2 "maskN" --alpha $2 --epoch $3 --suffix $suffix
  python main_cl.py --dataset $1 --lr 0.001 --aug1 "dropN" --aug2 "permE" --alpha $2 --epoch $3 --suffix $suffix
  python main_cl.py --dataset $1 --lr 0.001 --aug1 "dropN" --aug2 "subgraph" --alpha $2 --epoch $3 --suffix $suffix
  python main_cl.py --dataset $1 --lr 0.001 --aug1 "maskN" --aug2 "permE" --alpha $2 --epoch $3 --suffix $suffix
  python main_cl.py --dataset $1 --lr 0.001 --aug1 "maskN" --aug2 "subgraph" --alpha $2 --epoch $3 --suffix $suffix
  python main_cl.py --dataset $1 --lr 0.001 --aug1 "subgraph" --aug2 "permE" --alpha $2 --epoch $3 --suffix $suffix
done

