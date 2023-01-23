# CQD:
python3 main.py --cuda --do_train --do_test --data_path data/FB15k-237-q2b -n 0 --rank 1000 -lr 0.1 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen
# LitCQD
python3 main.py --cuda --do_train --do_test --data_path data/scripts/generated/FB15K-237_dummy_kblrn --use_attributes -n 0 --rank 1000 -lr 0.1 --attr_loss mae --alpha 0.5 --geo cqd-complexa --batch_size 1024 --test_batch_size 100 --train_times 100 --valid_epochs 10 --print_on_screen