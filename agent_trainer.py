class AgentTrainer:
    def __init__(self, env: CompilerGymEnv):
        self.env = env

    def make_training_env(self, *args) -> compiler_gym.envs.CompilerEnv:
        del args
        return CycleOverBenchmarks(self.env.make_env(), self.env.train_benchmarks)

    def train(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(include_dashboard=False, ignore_reinit_error=True)
        tune.register_env("compiler_gym", self.make_training_env)
        analysis = tune.run(
            PPOTrainer,
            checkpoint_at_end=True,
            stop={"episodes_total": 5},
            config={
                "seed": 0xCC,
                "num_workers": 1,
                "env": "compiler_gym",
                "rollout_fragment_length": 5,
                "train_batch_size": 5,
                "sgd_minibatch_size": 5,
            }
        )
        agent = PPOTrainer(
            env="compiler_gym",
            config={
                "num_workers": 1,
                "seed": 0xCC,
                "explore": False,
            },
        )
        checkpoint = analysis.get_best_checkpoint(
            metric="episode_reward_mean", 
            mode="max", 
            trial=analysis.trials[0]
        )
        agent.restore(checkpoint)
        return agent
