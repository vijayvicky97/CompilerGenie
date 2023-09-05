from imports_and_versions import *
from compiler_gym_env import CompilerGymEnv
from agent_evaluator import AgentEvaluator
from agent_trainer import AgentTrainer

if __name__ == "__main__":
    # Setup environment
    env_setup = CompilerGymEnv()
    env_setup.get_datasets()

    # Train agent
    trainer = AgentTrainer(env_setup)
    trainer.train()

    # Evaluate agent
    evaluator = AgentEvaluator(env_setup)
    val_rewards = evaluator.run_agent_on_benchmarks(env_setup.val_benchmarks)
    test_rewards = evaluator.run_agent_on_benchmarks_with_agent(agent, env_setup.test_benchmarks)
