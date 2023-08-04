python3 ../scripts/train_rl.py --env BabyAI-GoToLocal_Dynamics_Train-v0 \
       --test_env BabyAI-GoToLocal_Dynamics_Test-v0 --arch expert_filmcnn \
       --frames=100000000 --flag='FiLM on babyai++' --instr-dim=64 --seed=42 --film_d=8
