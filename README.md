# SUGRL: Simple Unsupervised Graph Representation Learning

This repository contains the reference code for the paper Simple Self-supervised Multiplx Graph Representation Learning 

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Testing](#test)
0. [Training](#train)

## Installation
pip install -r requirements.txt 

## Preparation

weights see >>>[here](SUGRL/checkpoints/)<<<.

Configs see >>>[here](SUGRL/args.yaml)<<<.

Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)



Important args:
* `--pretrain` Test checkpoints
* `--dataset-name` acm, imdb, dblp, freebase
* `--custom_key` Node: node classification  Clu: clustering   Sim: similarity

