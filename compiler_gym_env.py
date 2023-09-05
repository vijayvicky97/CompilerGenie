class CompilerGymEnv:
    def __init__(self):
        self.env = None
        self.train_benchmarks = []
        self.val_benchmarks = []
        self.test_benchmarks = []

    def make_env(self) -> compiler_gym.envs.CompilerEnv:
        env = compiler_gym.make(
            "llvm-v0",
            observation_space="Autophase",
            reward_space="IrInstructionCount",
        )
        env = TimeLimit(env, max_episode_steps=5)
        return env

    def get_datasets(self):
        with self.make_env() as env:
            npb = env.datasets["npb-v0"]
            chstone = env.datasets["chstone-v0"]
            train_benchmarks = list(islice(npb.benchmarks(), 55))
            self.train_benchmarks, self.val_benchmarks = train_benchmarks[:50], train_benchmarks[50:]
            self.test_benchmarks = list(chstone.benchmarks())
