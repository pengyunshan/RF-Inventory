cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models gbdt --output_dir experiments/gbdt_final --save_predictions --n_vis_contexts 2 2>&1

cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models tf_smoothed_minmax --output_dir experiments/tf_smoothed_minmax_onehot --save_predictions --n_vis_contexts 2 2>&1

cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models tf_minmax --output_dir experiments/tf_minmax_onehot --save_predictions --n_vis_contexts 2 2>&1

cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models tf_gcm --output_dir experiments/tf_gcm --save_predictions --n_vis_contexts 2 2>&1

cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models tf_hint --output_dir experiments/tf_hint --save_predictions --n_vis_contexts 2 2>&1

cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models tf_pwl --output_dir experiments/tf_pwl --save_predictions --n_vis_contexts 2 2>&1

cd /Users/pengyunshan/workspace/rf_dataset && python scripts/train_and_evaluate.py --train_path data/df_train3.pq --test_path data/df_test3.pq --split_type joint --models tf_posnn --output_dir experiments/tf_posnn --save_predictions --n_vis_contexts 2 2>&1

