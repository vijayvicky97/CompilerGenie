class AgentEvaluator:
    def __init__(self, env: CompilerGymEnv):
        self.env = env

    def run_agent_on_benchmarks(self, benchmarks, subsequence1=None, subsequence2=None):
        with self.env.make_env() as env:
            rewards = []
            prev_reward = None
            current_subsequence = subsequence1
            for seq_idx in range(len(current_subsequence)):
                for i, benchmark in enumerate(benchmarks, start=1):
                    observation, done = env.reset(benchmark=benchmark), False
                    action_idx = 0
                    while not done:
                        action = current_subsequence[seq_idx][action_idx]
                        observation, reward, done, _ = env.step(action)
                        if prev_reward is not None and reward < prev_reward:
                            current_subsequence = subsequence2 if current_subsequence == subsequence1 else subsequence1
                            action_idx = 0
                        action_idx = (action_idx + 1) % len(current_subsequence)
                        prev_reward = reward
                    rewards.append(env.episode_reward)
            return rewards

    def run_agent_on_benchmarks_with_agent(self, agent, benchmarks):
        with self.env.make_env() as env:
            rewards = []
            for i, benchmark in enumerate(benchmarks, start=1):
                observation, done = env.reset(benchmark=benchmark), False
                while not done:
                    action = agent.compute_action(observation)
                    observation, _, done, _ = env.step(action)
                rewards.append(env.episode_reward)
            return rewards
