from flask import Flask, jsonify, request
from compiler_gym_env import CompilerGymEnv
from agent_evaluator import AgentEvaluator
from agent_trainer import AgentTrainer

app = Flask(__name__)

env_setup = CompilerGymEnv()
env_setup.get_datasets()
trainer = AgentTrainer(env_setup)
evaluator = AgentEvaluator(env_setup)

@app.route('/train', methods=['POST'])
def train_agent():
    agent = trainer.train()
    return jsonify({"message": "Training completed!"})

@app.route('/evaluate', methods=['GET'])
def evaluate_agent():
    val_rewards = evaluator.run_agent_on_benchmarks(env_setup.val_benchmarks)
    test_rewards = evaluator.run_agent_on_benchmarks_with_agent(agent, env_setup.test_benchmarks)
    return jsonify({
        "val_rewards": val_rewards,
        "test_rewards": test_rewards
    })

if __name__ == '__main__':
    app.run(debug=True)
