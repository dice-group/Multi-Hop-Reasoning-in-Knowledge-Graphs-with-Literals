# Evaluation script for LitCQD: Multi-Hop Reasoning in Knowledge Graphs with Literals
# Evaluate performance for complex queries
# To reproduce the results reported in Table 2.
# CQD
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_no_attr --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen
# LitCQD
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/FB15k-237-q2b --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# To reproduce the results reported in Table 3 and Table 4.
python3 eval_cqd.py --do_test --checkpoint_path checkpoints_FB15K-237/checkpoint_orig_attr_kblrn --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes --rank 1000 --geo cqd-complexa --test_batch_size 1024 --print_on_screen

# To reproduce the results reported in Table 5.
python3 eval_attribute_filtering_example.py

