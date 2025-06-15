import argparse
import json
import time
from datetime import timedelta
from typing import Any, Dict

from llama_index.core.llms import LLM

from mage.agent import TopAgent
from mage.benchmark_read_helper import (
    TypeBenchmark,
    TypeBenchmarkFile,
    get_benchmark_contents,
)
from mage.gen_config import get_llm, set_exp_setting
from mage.log_utils import get_logger
from mage.sim_reviewer import sim_review_golden_benchmark
from mage.token_counter import TokenCount

logger = get_logger(__name__)


args_dict = {
    "provider": "anthropic",
    #"provider": "openai",
    "model": "claude-3-5-sonnet-20241022",
    # "model": "gemini-2.0-flash-001",
    # "model": "claude-3-7-sonnet-20250219",
    # "model": "gpt-4o-2024-08-06",
    # "filter_instance": "^(Prob070_ece241_2013_q2|Prob151_review2015_fsm)$",
    #"filter_instance": "^(Prob011_norgate)$",
    "filter_instance": "^(.*)$",
    "type_benchmark": "verilog_eval_v2",
    "path_benchmark": "verilog-eval",
    #"run_identifier": "gpt-4o-2024-08-06_run_at5",
    "run_identifier": "vanilla_rag_claude-3-5-sonnet-20241022",
    "n": 1,
    "temperature": 0.85,
    "top_p": 0.95,
    "max_token": 8192,
    "use_golden_tb_in_mage": True,
    "key_cfg_path": "./key.cfg",
    # Vanilla RAG toggles:
    "rag_csv_path": "Datasets/RTLCoder/rtl_coder_dataset.csv",
    "rag_k": 3,
}


def run_round(args: argparse.Namespace, llm: LLM):
    total_start_time = time.monotonic()
    type_benchmark = TypeBenchmark[args.type_benchmark.upper()]
    spec_dict = get_benchmark_contents(
        type_benchmark,
        TypeBenchmarkFile.SPEC,
        args.path_benchmark,
        args.filter_instance,
    )
    golden_tb_path_dict = get_benchmark_contents(
        type_benchmark,
        TypeBenchmarkFile.TEST_PATH,
        args.path_benchmark,
        args.filter_instance,
    )
    golden_rtl_path_dict = get_benchmark_contents(
        type_benchmark,
        TypeBenchmarkFile.GOLDEN_PATH,
        args.path_benchmark,
        args.filter_instance,
    )

    #agent = TopAgent(llm)
     
    agent = TopAgent(
        llm,
        rag_csv_path=args.rag_csv_path,
        rag_k=args.rag_k,
    )

    agent.set_output_path(f"./output_{args.run_identifier}")
    agent.set_log_path(f"./log_{args.run_identifier}")
    agent.set_redirect_log(True)
    # agent.set_ablation(True)
    record_file = f"./output_{args.run_identifier}/record.json"
    record_json: Dict[str, Dict[str, Any]] = {"record_per_run": {}, "total_record": {}}

    ret: dict[str, tuple[bool, str]] = {}
    review_result: dict[str, tuple[bool, str]] = {}
    pass_cnt = 0
    token_sum = TokenCount(in_token_cnt=0, out_token_cnt=0)
    token_limit_cnt = 0
    for i, (task_id, spec) in enumerate(spec_dict.items()):
        start_time = time.monotonic()
        print(f"({i+1:03d}/{len(spec_dict):03d}) Current task: {task_id}")
        ret[task_id] = agent.run(
            benchmark_type_name=type_benchmark.name,
            task_id=task_id,
            spec=spec,
            golden_tb_path=(
                golden_tb_path_dict[task_id] if args.use_golden_tb_in_mage else None
            ),
            golden_rtl_blackbox_path=(
                golden_rtl_path_dict[task_id] if args.use_golden_tb_in_mage else None
            ),
        )
        run_time = timedelta(seconds=time.monotonic() - start_time)
        print(f"{task_id} took {run_time} to execute")
        is_pass, golden_sim_log = sim_review_golden_benchmark(
            task_id=task_id,
            output_path=agent.output_path,
            benchmark_type=type_benchmark,
            benchmark_path=args.path_benchmark,
        )
        print(f"({i+1:03d}/{len(spec_dict):03d}) {task_id}: is_pass = {is_pass}")
        run_token_cnt = agent.token_counter.get_sum_count()
        print(
            f"Current problem token count: Input {run_token_cnt.in_token_cnt}, Output {run_token_cnt.out_token_cnt}"
        )
        if agent.token_counter.token_cost:
            run_cost = (
                run_token_cnt.in_token_cnt
                * agent.token_counter.token_cost.in_token_cost_per_token
                + run_token_cnt.out_token_cnt
                * agent.token_counter.token_cost.out_token_cost_per_token
            )
        run_token_limit_cnt = agent.token_counter.get_total_token()
        print(f"Current problem token limit consumption: {run_token_limit_cnt}")
        token_limit_cnt += run_token_limit_cnt
        print(f"{'Current problem token cost':<25}: ${run_cost:.2f} USD")
        token_sum += run_token_cnt
        pass_cnt += is_pass
        review_result[task_id] = (is_pass, golden_sim_log)
        record_json["record_per_run"][task_id] = {
            "is_pass": is_pass,
            "run_token_limit_cnt": f"{run_token_limit_cnt:.2f}",
            "run_token_cost": f"{run_cost:.2f}",
            "run_time": str(run_time),
        }
    print(f"Pass rate: {pass_cnt}/{len(spec_dict)}")
    print(
        f"Total token count: Input {token_sum.in_token_cnt}, Output {token_sum.out_token_cnt}"
    )
    print(f"Total token limit consumption: {token_limit_cnt}")
    if agent.token_counter.token_cost:
        total_cost = (
            token_sum.in_token_cnt
            * agent.token_counter.token_cost.in_token_cost_per_token
            + token_sum.out_token_cnt
            * agent.token_counter.token_cost.out_token_cost_per_token
        )
        print(f"{'Total cost':<25}: ${total_cost:.2f} USD")
        print(f"{'Avg cost':<25}: ${total_cost / len(spec_dict):.2f} USD")

    total_run_time = timedelta(seconds=time.monotonic() - total_start_time)
    print(f"Totally took {total_run_time} to execute")
    record_json["total_record"] = {
        "pass_cnt": pass_cnt,
        "total_cnt": len(spec_dict),
        "token_limit_cnt": token_limit_cnt,
        "total_cost": f"{total_cost:.2f}",
        "avg_cost": f"{total_cost / len(spec_dict):.2f}",
        "total_run_time": str(total_run_time),
    }
    json.dump(record_json, open(record_file, "w"), indent=4)


def main():
    args = argparse.Namespace(**args_dict)

    llm = get_llm(
        model=args.model,
        cfg_path=args.key_cfg_path,
        max_token=args.max_token,
        provider=args.provider,
    )
    identifier_head = args.run_identifier
    n = args.n
    set_exp_setting(temperature=args.temperature, top_p=args.top_p)

    for i in range(n):
        print(f"Round {i+1}/{n}")
        args.run_identifier = f"{identifier_head}_{i}"
        run_round(args, llm)


if __name__ == "__main__":
    main()
