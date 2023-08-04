python3 ../scripts/train_rl.py --env BabyAI-OpenTwoDoors-v0 --test_env BabyAI-OpenTwoDoors-v0 \
       --arch NPS_vanilla --seed=1 \
       --frames=3000000 \
       --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 --rule_codebook_size=16\
       --instr-dim=64 --rule_dim=64 --use_all_slots \
       --use_compositional_split --num_contexts=1 --act_dim=3 --compositional_step=1 \
       --memory-dim 1024 --procs 32 --flag='NPS vanilla with 1 context, 1204 hidden state size and 3e7 frames on open two doors'\
       --model "FILM_lstm_124" --memory-dim=1024 --arch="expert_filmcnn" 
