python3 ../scripts/train_rl.py --env BabyAI-GoToLocal-v0 --arch NPS_vanilla --num_contexts=1 \
        --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=8 \
        --frames=9000000 --use_all_slots --flag='Test-collapse' --memory-dim=130 --use_rim --concat_instr_to_mem \
        --log-interval=1 --save-interval=1 --val-episodes=9 --recurrence=20 --frames-per-proc=80 --free_rule_param \
        --sparse_features
