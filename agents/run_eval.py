import os
import argparse
import joblib
import json
import benchmark_run.utils as utils
from benchmark_run.Meta_agent_arch import get_agent
from benchmark_run.llms import get_llm_backend, reset_kv_cache
from benchmark_run.config import available_agent_names

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="ZeroshotThink_HotPotQA_run_Agent")
parser.add_argument("--plan_agent", type=str, help="Model name of the plan agent on the worker", default="plan")
parser.add_argument("--action_agent", type=str, help="Model name of the action agent on the worker", default="action")
parser.add_argument("--reflect_agent", type=str, help="Model name of the reflect agent on the worker", default="reflect")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=4096)
parser.add_argument("--task",type=str ,help="task name",default="Hotpotqa")
parser.add_argument("--task_path",type=str,help="task path")
parser.add_argument("--save_path",type=str,help="save path")
parser.add_argument("--trials", type=int, help="Complete passes over the benchmark", default=20)
parser.add_argument("--limit", type=int, help="Questions per level (default: all)", default=None)
args = parser.parse_args()

agent_name = args.agent_name

plan_agent = args.plan_agent
action_agent = args.action_agent
reflect_agent = args.reflect_agent
max_context_len = args.max_context_len
save_path = args.save_path
task_path = args.task_path
assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def run_one_complex_level_hotpotqa(level="easy", test_num=20):
    hotpot = joblib.load(f'{task_path}/{level}.joblib').reset_index(drop = True)
    for i in range(test_num):
        print(f"{i} / {test_num} test")
        agent_save_file = f"{save_path}/{level}/{i}.jsonl"
        task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()][:args.limit]
        if os.path.exists(agent_save_file):
            sessions = utils.get_all_agent_sessions(agent_save_file)
            completed_tasks = utils.get_non_error_tasks(sessions)
            print(f"{level}:{len(completed_tasks)}")
            task_instructions = [task for task in task_instructions if task not in completed_tasks]
            utils.delete_error(agent_save_file)

        llm_plan = get_llm_backend(plan_agent)
        llm_action = get_llm_backend(action_agent)
        llm_reflect = get_llm_backend(reflect_agent)

        agent_cls = get_agent(agent_name)
        agents = [agent_cls(ques, ans, llm_plan.run, llm_action.run, llm_reflect.run, max_context_len) for ques, ans in task_instructions]
        for agent in agents:
            process_agent_run_step(agent)
            reset_kv_cache()
            utils.log_agent(agent, agent_save_file)
        print(f'Finished Trial {level}. Total: {len(agents)}')

def run_one_complex_level_scienceqa(level="1-4", test_num=20):
    scienceqa = json.load(open(f'{task_path}/format_scienceqa_grade{level}.json'))
    for i in range(test_num):
        print(f"{i} / {test_num} test")
        agent_save_file = f"{save_path}/{level}/{i}.jsonl"
        task_instructions = [(row['Question'],row['choices'],row['Answer'],row['orc'],row['caption']) for row in scienceqa][:args.limit]
        if os.path.exists(agent_save_file):
            sessions = utils.get_all_agent_sessions(agent_save_file)
            completed_tasks = utils.get_non_error_tasks(sessions)
            print(f"{level}:{len(completed_tasks)}")
            task_instructions = [task for task in task_instructions if task[0] not in completed_tasks]
            utils.delete_error(agent_save_file)

        llm_plan = get_llm_backend(plan_agent)
        llm_action = get_llm_backend(action_agent)
        llm_reflect = get_llm_backend(reflect_agent)

        agent_cls = get_agent(agent_name)
        agents = [agent_cls(ques, choices, ans, caption, orc, llm_plan.run, llm_action.run, llm_reflect.run, max_context_len) for ques,choices,ans,orc,caption in task_instructions]
        for agent in agents:
            process_agent_run_step(agent)
            reset_kv_cache()
            utils.log_agent(agent, agent_save_file)
        print(f'Finished Trial. Total: {len(agents)}')

def main():
    test_num = args.trials
    reset_kv_cache()
    if args.task == "Hotpotqa":
        levels = ['hard', 'medium', 'easy']
        for level in levels:
            os.makedirs(f"{save_path}/{level}", exist_ok=True)
            run_one_complex_level_hotpotqa(level, test_num)
    elif args.task == "Scienceqa":
        levels = ['9-12', '5-8', '1-4']
        for level in levels:
            os.makedirs(f"{save_path}/{level}", exist_ok=True)
            run_one_complex_level_scienceqa(level, test_num)
if __name__ == '__main__':
    main()
