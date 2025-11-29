import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent.graph_hybrid import run_agent

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

# agent = RetailAnalyticsCopilot()

with open(args.batch) as f:
    questions = [json.loads(line) for line in f]

outputs = []
for q in questions:
    result = run_agent(q["question"], q["format_hint"])
    result["id"] = q["id"]
    outputs.append(result)

with open(args.out, "w") as f:
    for line in outputs:
        f.write(json.dumps(line) + "\n")

print(f"Results saved to {args.out}")


