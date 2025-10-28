# eval_kpis.py - Run after 50+ queries to compute real KPIs
import json
import glob
from statistics import mean

# Load all metrics files
metrics_files = glob.glob("logs/metrics_*.json")
kpis = {
    "retrieval_precision": [],
    "grounding_fidelity": [],
    "task_completion": [],
    "latency_total": [],
    "automation_depth": []
}

for file in metrics_files:
    with open(file, 'r') as f:
        data = json.load(f)
        kpis["retrieval_precision"].append(data.get("retrieval_precision", 0))
        kpis["grounding_fidelity"].append(data.get("grounding_fidelity", 0))
        kpis["task_completion"].append(data.get("task_completion", 0))
        kpis["latency_total"].append(data.get("latency_total", 0))
        kpis["automation_depth"].append(data.get("automation_depth", 0))

# Compute averages
report = {
    "avg_retrieval_precision": mean(kpis["retrieval_precision"]),
    "avg_grounding_fidelity": mean(kpis["grounding_fidelity"]),
    "avg_task_completion": mean(kpis["task_completion"]),
    "avg_latency": mean(kpis["latency_total"]),
    "avg_automation_depth": mean(kpis["automation_depth"]),
    "total_runs": len(metrics_files)
}

print("RAGent KPI Report:")
print(json.dumps(report, indent=2))
with open("kpi_report.json", "w") as f:
    json.dump(report, f, indent=2)