# SSMGRL: Simple Self-supervised Multiplex Graph Representation Learning

This repository contains the reference code for the paper Simple Self-supervised Multiplx Graph Representation Learning 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Training](#train)
0. [Testing](#test)

## Installation
pip install -r requirements.txt 

## Preparation

To train the model, unzip the dataset first.

Important args:
* `--use_pretrain` Test checkpoints
* `--dataset` acm, imdb, dblp, freebase
* `--custom_key` Node: node classification  Clu: clustering   Sim: similarity

## Training
python main.py

## Test
Choose the custom_key of different downstream tasks
