
# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --no-instr --num_contexts=1 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on babyai++, using all slots'

# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --num_contexts=1 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on babyai++, using all slots' --instr-arch='gru'

python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch NPS_vanilla --num_contexts=1 --seed=42\
       --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on babyai, using all slots' --no-instr --num_rules=8 --instr-dim=64 --rule_dim=64
       
# python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 --test_env BabyAI-GoToLocal_Dynamics_Train-v0 --arch NPS_vanilla --num_contexts=1 \
#        --frames=20000000 --use_all_slots --flag='NPS vanilla with 1 context and 2e7 frames on babyai++, using all slots' --instr-arch='gru'
