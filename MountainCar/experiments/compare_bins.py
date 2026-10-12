from src.train import train
from src.evaluate import evaluate

CONFIGURATIONS = [(10, 10), (20, 20), (40, 40), (20, 40), (40, 20)]
SEEDS = [42, 123, 456]
for config in CONFIGURATIONS:
    for seed in SEEDS[2:3]:
        # train(config, seed)
        average_reward, success_rate = evaluate(config, seed)

        print(f"Configuration: {config}")
        print(f"Average reward: {average_reward:.2f}")
        print(f"Success rate: {success_rate:.1f}%")