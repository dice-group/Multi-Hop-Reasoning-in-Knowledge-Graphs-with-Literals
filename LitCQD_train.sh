# CQD:
python3 main.py --cuda --do_train --do_test --data_path data/FB15k-237-q2b -n 0 --rank 1000 -lr 0.1 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen
(litcqd) cdemir@raki:~/Multi-Hop-Reasoning-in-Knowledge-Graphs-with-Literals$ python3 main.py --do_train --do_test --data_path data/FB15k-237-q2b -n 0 --rank 1000 -lr 0.1 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen
Namespace(cqd_params=CQDParams(cqd_type=<type.discrete: 2>, cqd_t_norm=<t-norm.prod: 1>, cqd_k=4), dataloader_type='cpp', hyperparams=HyperParams(rank=1000, batch_size=1024, attr_loss=<attr_loss.mae: 1>, learning_rate=0.1, learning_rate_attr=0.1, negative_sample_size=0, negative_attr_sample_size=0, reg_weight=0, reg_weight_ent=0, reg_weight_rel=0, reg_weight_attr=0, alpha=0.3, optimizer=<optimizer.adagrad: 2>, scheduler_patience=5, scheduler_factor=0.95, scheduler_threshold=0.01, margin=2.0, p_norm=2, do_sigmoid=False, rank_attr=50, desc_emb=<desc_emb.1-layer: 1>, use_modulus=False), print_on_screen=True, train_config=TrainConfig(data_path='data/FB15k-237-q2b', save_path=None, checkpoint_path=None, geo=<geo.cqd-complexa: 8>, loss=<loss.ce: 2>, train_times=100, valid_epochs=10, cpu_num=10, seed=0, cuda=False, use_attributes=False, use_descriptions=False, train_data_type=<train_data_type.triples: 2>, test_batch_size=100, do_tune=False, do_train=True, do_test=True, eval_on_train=False, simple_eval=False, word_emb_dim=300))
logging to logs/FB15k-237-q2b/cqd-complexa/2023.02.22-10:14:35
2023-02-22 10:14:51,654 INFO     train: 1p: 149689
2023-02-22 10:14:54,306 INFO     valid: 1p: 20101
2023-02-22 10:14:57,168 INFO     valid: 1p: 20101
2023-02-22 10:14:57,168 INFO     valid: 1dp: 0
2023-02-22 10:14:57,168 INFO     valid: di: 0
2023-02-22 10:14:57,682 INFO     ---------------------------------------------------------------------------------------------
2023-02-22 10:14:57,682 INFO     Geo: geo.cqd-complexa
2023-02-22 10:14:57,682 INFO     Data Path: None
2023-02-22 10:14:57,682 INFO     #entity: 14505
2023-02-22 10:14:57,682 INFO     #relation: 474
2023-02-22 10:14:57,682 INFO     #attributes: 0
2023-02-22 10:14:57,682 INFO     batch size: 1024
2023-02-22 10:14:57,715 INFO     Model Parameter Configuration:
2023-02-22 10:14:57,716 INFO     Parameter ent_embeddings.weight: torch.Size([14505, 2000]), require_grad = True
2023-02-22 10:14:57,716 INFO     Parameter rel_embeddings.weight: torch.Size([474, 2000]), require_grad = True
2023-02-22 10:14:57,716 INFO     Parameter Number: 29958000
2023-02-22 10:14:57,716 INFO     Ramdomly Initializing cqd-complexa Model...
  0%|                                                                                                                                                                              | 0/100 [00:00<?, ?it/s]Aborted
(litcqd) cdemir@raki:~/Multi-Hop-Reasoning-in-Knowledge-Graphs-with-Literals$ python3 main.py --do_train --do_test --data_path data/FB15k-237-q2b -n 0 --rank 1 -lr 0.1 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen
Namespace(cqd_params=CQDParams(cqd_type=<type.discrete: 2>, cqd_t_norm=<t-norm.prod: 1>, cqd_k=4), dataloader_type='cpp', hyperparams=HyperParams(rank=1, batch_size=1024, attr_loss=<attr_loss.mae: 1>, learning_rate=0.1, learning_rate_attr=0.1, negative_sample_size=0, negative_attr_sample_size=0, reg_weight=0, reg_weight_ent=0, reg_weight_rel=0, reg_weight_attr=0, alpha=0.3, optimizer=<optimizer.adagrad: 2>, scheduler_patience=5, scheduler_factor=0.95, scheduler_threshold=0.01, margin=2.0, p_norm=2, do_sigmoid=False, rank_attr=50, desc_emb=<desc_emb.1-layer: 1>, use_modulus=False), print_on_screen=True, train_config=TrainConfig(data_path='data/FB15k-237-q2b', save_path=None, checkpoint_path=None, geo=<geo.cqd-complexa: 8>, loss=<loss.ce: 2>, train_times=100, valid_epochs=10, cpu_num=10, seed=0, cuda=False, use_attributes=False, use_descriptions=False, train_data_type=<train_data_type.triples: 2>, test_batch_size=100, do_tune=False, do_train=True, do_test=True, eval_on_train=False, simple_eval=False, word_emb_dim=300))
logging to logs/FB15k-237-q2b/cqd-complexa/2023.02.22-10:15:38
2023-02-22 10:15:54,841 INFO     train: 1p: 149689
2023-02-22 10:15:57,484 INFO     valid: 1p: 20101
2023-02-22 10:16:00,342 INFO     valid: 1p: 20101
2023-02-22 10:16:00,343 INFO     valid: 1dp: 0
2023-02-22 10:16:00,343 INFO     valid: di: 0
2023-02-22 10:16:00,672 INFO     ---------------------------------------------------------------------------------------------
2023-02-22 10:16:00,672 INFO     Geo: geo.cqd-complexa
2023-02-22 10:16:00,672 INFO     Data Path: None
2023-02-22 10:16:00,672 INFO     #entity: 14505
2023-02-22 10:16:00,672 INFO     #relation: 474
2023-02-22 10:16:00,672 INFO     #attributes: 0
2023-02-22 10:16:00,672 INFO     batch size: 1024
2023-02-22 10:16:00,708 INFO     Model Parameter Configuration:
2023-02-22 10:16:00,709 INFO     Parameter ent_embeddings.weight: torch.Size([14505, 2]), require_grad = True
2023-02-22 10:16:00,709 INFO     Parameter rel_embeddings.weight: torch.Size([474, 2]), require_grad = True
2023-02-22 10:16:00,709 INFO     Parameter Number: 29958
2023-02-22 10:16:00,709 INFO     Ramdomly Initializing cqd-complexa Model...
  0%|                                                                                                                                                                              | 0/100 [00:00<?, ?it/s]Aborted


# LitCQD # Doesnt work
python3 main.py --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

python3 main.py --do_tune --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen

