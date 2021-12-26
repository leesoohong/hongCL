#!/bin/bash -ex

for suffix in 0 1 2 3 4 
do
  python main.py --dataset $1 --lr 0.001 --aug1 "maskN" --aug2 "subgraph" --alpha $2 --epoch $3 --suffix $suffix
done

