# LitCQD: Multi-Hop Reasoning in Knowledge Graphs with Literals

The implementation is based on the publicly available implementation of Query2Box ([Link](https://github.com/snap-stanford/KGReasoning)) .
To load the dataset and the pretrained models, please download, unzip the two files
checkpoints_FB15K-237 and data from
([GoogleDriveLink](https://drive.google.com/drive/folders/1qXwWWlNO84Y1s8O0_1S1BDMXbv8MK3LK?usp=sharing)) into the current working directory, i.e., here.

## Installation
```
conda create -n temp python=3.8
pip3 install torch==1.9.0 --find-links https://download.pytorch.org/whl/torch_stable.html
pip3 install bidict==0.21.3
pip3 install gensim==4.1.2
pip3 install ray[tune]==1.9.1
pip3 install simple-parsing==0.0.17
pip3 install tqdm==4.62.0
pip3 install tensorboardX==2.4.1
pip3 install tensorboard==2.7.0
```

## Reproduce Reported Results
Executing the following command results in evaluating the  LitCQD on FB15K-237 dataset provided within KBLRN
```
conda activate temp
sh LitCQD_results.sh
```
Please refer to Table 1 in our manuscript for different query types ap,2ap,3ap,ai-lt,ai-eq,ai-gt,2ai,aip,pai,au

## Datasets

- `data/FB15k-237-q2b` contains the FB15K-237 dataset including generated queries provided by the authors of the Query2Box paper.
- `data/scripts/data`:
    - `numeric`: The numeric attribute data from different sources.
    - `relational`: The relational data for the FB15K-237 and LitWD1K datasets.
    - `textual`: The textual descriptions and entity names form different sources.
    

The FB15K-237 data provided by Query2Box is described as the following by the authors:
- `train.txt/valid.txt/test.txt`: KG edges
- `id2rel/rel2id/ent2id/id2ent.pkl`: KG entity relation dicts
- `train-queries/valid-queries/test-queries.pkl`: `defaultdict(set)`, each key represents a query structure, and the value represents the instantiated queries
- `train-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`)
- `valid-easy-answers/test-easy-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) / valid graph (edges in `train.txt`+`valid.txt`)
- `valid-hard-answers/test-hard-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the **additional** answers obtained in the validation graph (edges in `train.txt`+`valid.txt`) / test graph (edges in `train.txt`+`valid.txt`+`test.txt`)


### Query Generation

To generate the complex queries in `data/scripts/generated`, the python script `data/scripts/main.py` is used.
It requires yaml configuration files. For example, using the following configuration, the script generates queries for the FB15K-237 dataset and the attribute and descriptions data provided by LiteralE:
```yaml
input: 
  name: PickleMappings # rel2id/ent2id mappings are stored as pickle files
  header: False # triples csv file does not have a header line
  path: data/relational/FB15K-237
  has_inverse: True # contains reciprocal relations
  add_inverse: False # add reciprocal relations
  add_attr_exists_rel: False # add facts representing the existence of an attribute
literals:
  name: literale
  path: data/numeric/LiteralE
  normalize: True # apply min-max normalization
  valid_size: 1000
  test_size: 1000
descriptions:
  path: data/textual/LiteralE
  name_path: data/textual/DKRL
  google_news: True # use word embeddings based on Google News or use self-trained
  jointly: False # Generated queries for joint learning
  valid_size: 1000
  test_size: 1000
output:
  name: file # store output as csv files
  path: test_output
  queries:
    generate: True # generate complex queries
    path: test_output
    print_debug: True
    complex_train_queries: False # generate complex queries for training data; required by Query2Box
```
