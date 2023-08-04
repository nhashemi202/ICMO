python3 ../scripts/train_rl.py --env BabyAI-SynthS5R2-v0 \
--test_env BabyAI-SynthS5R2-v0 \
--seed=1 \
--frames=300000000 \
--instr_to_rule_mode='linear' --instr_arch='gru' --num_rules=8 --rule_codebook_size=16 \
--instr_dim=64 --rule_dim=64 \
--num_contexts=1 --act_dim=1 --compositional_step=1 \
--memory_dim 1024 \
--model "FiLM" --arch='expert_filmcnn' \
--procs=16 --batch_size=1280 --frames_per_proc=80 --recurrence=20 --rim_slowness_factor=4 \
--save_interval=500 \
#--use_all_slots \
--use_compositional_split \
# --input_mem_concat_instr \
# --use_hidden_feedback_contextual
