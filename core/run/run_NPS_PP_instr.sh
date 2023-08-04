# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --num_contexts=1 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='linear' --instr-arch='gru' --num_rules=12 \
#         --frames=9000000 --use_all_slots --flag='NPS vanilla using all slots and instructions gru single linear'

### bash run_NPS_PP_instr.sh instr_dim \\ num_rules \\ num_contexts \\ seed \\ instr_to_rule_mode \\ instr_arch \\ rule_codebook_size
python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --num_contexts=$3 \
        --instr-dim=$1 --rule_dim=$1 --instr_to_rule_mode=$5 --instr-arch=$6 --num_rules=$2 --seed=$4 --rule_codebook_size=$7 --use_large_model \
        --frames=20000000 --use_all_slots --flag='NPS vanilla using all slots and instructions gru single linear'

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --num_contexts=3 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='VQ' --instr-arch='VQ' --num_rules=8 --rule_codebook_size=16 --seed=42 \
#         --frames=20000000 --use_all_slots --flag='NPS vanilla using all slots and instructions gru single linear'

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --num_contexts=3 \
#         --instr-dim=64 --rule_dim=64 --instr_to_rule_mode='MHA' --instr-arch='MHA' --num_rules=8 --rule_codebook_size=16 --seed=42 \
#         --frames=20000000 --use_all_slots --flag='NPS vanilla using all slots and instructions gru single linear'


#### bash run_NPS_PP_instr.sh 64 8 1 1 MHA MHA 16
