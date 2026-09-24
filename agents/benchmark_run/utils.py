"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import json
import os


def log_agent(agent, file_path):
    save_dict = {"question": agent.question, "answer": agent.key, "correct": agent.is_correct(),
                 "reward": agent.reward()[0], "halted": agent.is_halted(), "error": agent.run_error,
                 "prompt": agent._build_agent_prompt()}
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as f:
        json.dump(save_dict, f)
        f.write("\n")


def get_all_agent_sessions(file_name):
    with open(file_name) as f:
        return [json.loads(line) for line in f]


def get_non_error_tasks(sessions):
    return list({(sess["question"], sess["answer"]) for sess in sessions if not sess["error"]})


def delete_error(file_name):
    sessions = get_all_agent_sessions(file_name)
    with open(file_name + '.back', 'a') as b_f:
        for sess in sessions:
            json.dump(sess, b_f)
            b_f.write('\n')
    with open(file_name, 'w') as f:
        for sess in sessions:
            if not sess["error"]:
                json.dump(sess, f)
                f.write('\n')


SEARCH_NOT_CONFIGURED = (
    "Web search is not configured. Implement call_web_search() in "
    "agents/benchmark_run/utils.py with your own search API."
)


def call_web_search(query, count: int = 5):
    """Search snippets for `query`; plug in your own search API here."""
    return [SEARCH_NOT_CONFIGURED]
