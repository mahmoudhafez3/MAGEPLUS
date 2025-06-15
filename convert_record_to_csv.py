import json
import csv
import os

#run_id = "claude-3-5-sonnet-20241022_run_0"
run_id = "vanilla_rag_claude-3-5-sonnet-20241022_0"  
input_path = f"output_{run_id}/record.json"
output_path = f"output_{run_id}/summary_{run_id}.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"record.json not found at: {input_path}")

with open(input_path, "r") as f:
    data = json.load(f)

rows = []
for task_id, record in data["record_per_run"].items():
    row = {
        "task_id": task_id,
        "is_pass": record["is_pass"],
        "run_token_limit_cnt": record["run_token_limit_cnt"],
        "run_token_cost": record["run_token_cost"],
        "run_time": record["run_time"],
    }
    rows.append(row)

# Add summary row
summary = data["total_record"]
summary_row = {
    "task_id": "TOTAL",
    "is_pass": f"{summary['pass_cnt']}/{summary['total_cnt']}",
    "run_token_limit_cnt": summary["token_limit_cnt"],
    "run_token_cost": summary["total_cost"],
    "run_time": summary["total_run_time"],
}
rows.append(summary_row)

# Write CSV
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ summary.csv saved at: {output_path}")
