import os
import re
import json
import argparse
from collections import defaultdict


DEFAULT_SUITE_NAMES = [
    "libero_goal",
    "libero_spatial",
    "libero_10",
    "libero_object",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_suite_name(suite_name):
    if suite_name is None:
        return None
    return str(suite_name).strip()


def normalize_task_name(name):
    """
    用于名字兜底匹配。

    task_info 里可能是：
        KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1

    summary 里可能是：
        turn_on_the_stove_and_put_the_moka_pot_on_it_table_1

    这个函数会去掉前面的场景前缀：
        KITCHEN_SCENE3_
        LIVING_ROOM_SCENE2_
        STUDY_SCENE1_
        等等
    """

    if name is None:
        return None

    name = str(name).strip()
    name = re.sub(r"^[A-Z_]+_SCENE\d+_", "", name)

    return name.strip()


def load_task_info_mapping(task_info_json):
    """
    读取总任务描述 json。

    输入格式示例：

    {
        "libero_10": [
            {
                "id": 1,
                "name": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1",
                "category": "Background Textures",
                "difficulty_level": 4
            }
        ]
    }

    返回：
        id_map[(suite_name, task_info_id)] = task_info
        name_map[(suite_name, normalized_task_name)] = task_info
    """

    data = load_json(task_info_json)

    if not isinstance(data, dict):
        raise ValueError("task_info_json 应该是一个 dict，key 是 suite 名称，value 是任务列表。")

    id_map = {}
    name_map = {}

    for suite_name, task_list in data.items():
        suite_name = normalize_suite_name(suite_name)

        if not isinstance(task_list, list):
            print(f"[Warning] {suite_name} 的 value 不是 list，跳过。")
            continue

        for item in task_list:
            if not isinstance(item, dict):
                continue

            raw_task_id = item.get("id")
            raw_task_name = item.get("name")

            if raw_task_id is None:
                print(f"[Warning] task_info 中缺少 id: suite={suite_name}, item={item}")
                continue

            try:
                task_id = int(raw_task_id)
            except Exception:
                print(f"[Warning] 非法 task_info id: suite={suite_name}, id={raw_task_id}")
                continue

            task_info = {
                "id": task_id,
                "name": raw_task_name,
                "normalized_name": normalize_task_name(raw_task_name),
                "category": item.get("category", "unknown"),
                "difficulty_level": item.get("difficulty_level", None),
            }

            id_key = (suite_name, task_id)
            id_map[id_key] = task_info

            norm_name = task_info["normalized_name"]
            if norm_name is not None:
                name_key = (suite_name, norm_name)

                if name_key in name_map:
                    print(
                        f"[Warning] 归一化任务名重复，后一个会覆盖前一个: "
                        f"suite={suite_name}, norm_name={norm_name}"
                    )

                name_map[name_key] = task_info

    return id_map, name_map


def collect_summary_files(result_root, suite_names):
    """
    遍历 result_root 下四个 suite 的所有子目录，寻找 summary.json。

    目录结构：
        result_root/
            libero_goal/
                xxx/
                    summary.json
            libero_spatial/
                xxx/
                    summary.json
            libero_10/
                xxx/
                    summary.json
            libero_object/
                xxx/
                    summary.json
    """

    summary_paths = []

    for suite_name in suite_names:
        suite_name = normalize_suite_name(suite_name)
        suite_dir = os.path.join(result_root, suite_name)

        if not os.path.isdir(suite_dir):
            print(f"[Warning] suite 目录不存在: {suite_dir}")
            continue

        for sub_name in sorted(os.listdir(suite_dir)):
            sub_dir = os.path.join(suite_dir, sub_name)

            if not os.path.isdir(sub_dir):
                continue

            summary_path = os.path.join(sub_dir, "summary.json")

            if os.path.isfile(summary_path):
                summary_paths.append(summary_path)
            else:
                print(f"[Warning] 没有找到 summary.json: {sub_dir}")

    return summary_paths


def infer_suite_from_path(summary_path, result_root):
    rel_path = os.path.relpath(summary_path, result_root)
    parts = rel_path.split(os.sep)

    if len(parts) >= 1:
        return normalize_suite_name(parts[0])

    return "unknown_suite"


def match_task_info(
    suite_name,
    summary_task_id,
    summary_task_name,
    id_map,
    name_map,
):
    """
    匹配顺序：

    1. 用 id + 1 精确匹配：
        task_info_id = summary_task_id + 1
        id_map[(suite_name, task_info_id)]

    2. 如果 id + 1 匹配失败，用名字兜底：
        normalize_task_name(summary_task_name)
        name_map[(suite_name, normalized_name)]
    """

    task_info_id = summary_task_id + 1

    # 第一优先级：所有任务都按照 summary_id + 1 匹配 task_info 的 id
    task_info = id_map.get((suite_name, task_info_id))

    if task_info is not None:
        return task_info, "id_plus_1_match", task_info_id

    # 第二优先级：名字去前缀兜底
    norm_summary_name = normalize_task_name(summary_task_name)

    if norm_summary_name is not None:
        task_info = name_map.get((suite_name, norm_summary_name))

        if task_info is not None:
            return task_info, "name_prefix_fallback", task_info.get("id")

    return None, "unmatched", task_info_id


def analyze(result_root, task_info_json, suite_names):
    id_map, name_map = load_task_info_mapping(task_info_json)
    summary_paths = collect_summary_files(result_root, suite_names)

    stats_by_category = defaultdict(lambda: {
        "num_tasks": 0,
        "num_success": 0,
        "success_rate_sum": 0.0,
        "suites": set(),
        "match_methods": defaultdict(int),
        "difficulty_levels": defaultdict(lambda: {
            "num_tasks": 0,
            "num_success": 0,
            "success_rate_sum": 0.0,
        }),
        "tasks": [],
    })

    unmatched_tasks = []
    bad_summary_files = []

    for summary_path in summary_paths:
        try:
            summary = load_json(summary_path)
        except Exception as e:
            bad_summary_files.append((summary_path, str(e)))
            continue

        suite_name = summary.get("task_suite_name")

        if suite_name is None:
            suite_name = infer_suite_from_path(summary_path, result_root)

        suite_name = normalize_suite_name(suite_name)

        raw_summary_task_id = summary.get("task_id")
        summary_task_name = summary.get("task_name")

        if raw_summary_task_id is None:
            print(f"[Warning] summary 中缺少 task_id: {summary_path}")
            continue

        try:
            summary_task_id = int(raw_summary_task_id)
        except Exception:
            print(
                f"[Warning] 非法 summary task_id: "
                f"{summary_path}, task_id={raw_summary_task_id}"
            )
            continue

        try:
            success_rate = float(summary.get("success_rate", 0.0))
        except Exception:
            print(f"[Warning] 非法 success_rate: {summary_path}，设为 0.0")
            success_rate = 0.0

        task_info, match_method, matched_task_info_id = match_task_info(
            suite_name=suite_name,
            summary_task_id=summary_task_id,
            summary_task_name=summary_task_name,
            id_map=id_map,
            name_map=name_map,
        )

        if task_info is None:
            category = "unknown"
            difficulty_level = None
            task_info_name = None

            unmatched_tasks.append({
                "suite": suite_name,
                "summary_task_id": summary_task_id,
                "expected_task_info_id": summary_task_id + 1,
                "summary_task_name": summary_task_name,
                "normalized_summary_name": normalize_task_name(summary_task_name),
                "summary_path": summary_path,
            })
        else:
            category = task_info.get("category", "unknown")
            difficulty_level = task_info.get("difficulty_level", None)
            task_info_name = task_info.get("name")

        is_success = 1 if success_rate >= 1.0 else 0

        stats_by_category[category]["num_tasks"] += 1
        stats_by_category[category]["num_success"] += is_success
        stats_by_category[category]["success_rate_sum"] += success_rate
        stats_by_category[category]["suites"].add(suite_name)
        stats_by_category[category]["match_methods"][match_method] += 1

        if difficulty_level is not None:
            diff_key = str(difficulty_level)
            stats_by_category[category]["difficulty_levels"][diff_key]["num_tasks"] += 1
            stats_by_category[category]["difficulty_levels"][diff_key]["num_success"] += is_success
            stats_by_category[category]["difficulty_levels"][diff_key]["success_rate_sum"] += success_rate

        stats_by_category[category]["tasks"].append({
            "suite": suite_name,
            "summary_task_id": summary_task_id,
            "matched_task_info_id": matched_task_info_id,
            "summary_task_name": summary_task_name,
            "task_info_name": task_info_name,
            "normalized_summary_name": normalize_task_name(summary_task_name),
            "category": category,
            "difficulty_level": difficulty_level,
            "success_rate": success_rate,
            "success": is_success,
            "match_method": match_method,
            "summary_path": summary_path,
        })

    return stats_by_category, unmatched_tasks, bad_summary_files


def print_category_results(stats_by_category):
    print("\n" + "=" * 120)
    print("Success Rate by Perturbation Category")
    print("=" * 120)

    print(
        f"{'Category':<35} "
        f"{'Tasks':>8} "
        f"{'Success':>8} "
        f"{'Success Rate':>15} "
        f"{'Suites'}"
    )
    print("-" * 120)

    for category, item in sorted(stats_by_category.items(), key=lambda x: x[0]):
        num_tasks = item["num_tasks"]
        num_success = item["num_success"]
        avg_success_rate = item["success_rate_sum"] / num_tasks if num_tasks > 0 else 0.0
        suites = ", ".join(sorted(list(item["suites"])))

        print(
            f"{category:<35} "
            f"{num_tasks:>8} "
            f"{num_success:>8} "
            f"{avg_success_rate * 100:>14.2f}% "
            f"{suites}"
        )

    print("-" * 120)

    total_tasks = sum(v["num_tasks"] for v in stats_by_category.values())
    total_success = sum(v["num_success"] for v in stats_by_category.values())
    total_success_rate_sum = sum(v["success_rate_sum"] for v in stats_by_category.values())

    overall_rate = total_success_rate_sum / total_tasks if total_tasks > 0 else 0.0

    print(
        f"{'Overall':<35} "
        f"{total_tasks:>8} "
        f"{total_success:>8} "
        f"{overall_rate * 100:>14.2f}%"
    )

    print("=" * 120)


def print_match_method_results(stats_by_category):
    print("\n" + "=" * 120)
    print("Match Method Statistics")
    print("=" * 120)

    total_methods = defaultdict(int)

    for _, item in stats_by_category.items():
        for method, count in item["match_methods"].items():
            total_methods[method] += count

    for method, count in sorted(total_methods.items(), key=lambda x: x[0]):
        print(f"{method:<25} {count}")

    print("=" * 120)


def print_difficulty_results(stats_by_category):
    print("\n" + "=" * 120)
    print("Success Rate by Category and Difficulty Level")
    print("=" * 120)

    for category, item in sorted(stats_by_category.items(), key=lambda x: x[0]):
        print(f"\n[{category}]")

        difficulty_levels = item["difficulty_levels"]

        if not difficulty_levels:
            print("  No difficulty_level information.")
            continue

        print(
            f"  {'Difficulty':<12} "
            f"{'Tasks':>8} "
            f"{'Success':>8} "
            f"{'Success Rate':>15}"
        )
        print("  " + "-" * 55)

        for diff, diff_item in sorted(
            difficulty_levels.items(),
            key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0],
        ):
            num_tasks = diff_item["num_tasks"]
            num_success = diff_item["num_success"]
            avg_success_rate = diff_item["success_rate_sum"] / num_tasks if num_tasks > 0 else 0.0

            print(
                f"  {diff:<12} "
                f"{num_tasks:>8} "
                f"{num_success:>8} "
                f"{avg_success_rate * 100:>14.2f}%"
            )


def print_detail_results(stats_by_category):
    print("\n" + "=" * 120)
    print("Detailed Task Results")
    print("=" * 120)

    for category, item in sorted(stats_by_category.items(), key=lambda x: x[0]):
        print(f"\n[{category}]")

        tasks = sorted(
            item["tasks"],
            key=lambda x: (
                x["suite"],
                x["summary_task_id"] if x["summary_task_id"] is not None else -1,
            ),
        )

        for t in tasks:
            status = "SUCCESS" if t["success"] == 1 else "FAIL"

            print(
                f"  [{status}] "
                f"[{t['suite']}] "
                f"summary_id={t['summary_task_id']} "
                f"matched_info_id={t['matched_task_info_id']} "
                f"diff={t['difficulty_level']} "
                f"sr={t['success_rate']} "
                f"match={t['match_method']} "
                f"name={t['summary_task_name']}"
            )


def print_warnings(unmatched_tasks, bad_summary_files):
    if unmatched_tasks:
        print("\n" + "=" * 120)
        print("[Warning] 有些 summary 任务没有在 task_info_json 中找到。")
        print(f"数量: {len(unmatched_tasks)}")
        print("这些任务会被统计到 category = unknown。")
        print("\n前 100 个 unmatched tasks:")

        for item in unmatched_tasks[:100]:
            print(
                f"  - [{item['suite']}] "
                f"summary_id={item['summary_task_id']} "
                f"expected_info_id={item['expected_task_info_id']} "
                f"name={item['summary_task_name']} "
                f"norm={item['normalized_summary_name']}"
            )

    if bad_summary_files:
        print("\n" + "=" * 120)
        print("[Warning] 有些 summary.json 读取失败。")
        print(f"数量: {len(bad_summary_files)}")
        print("\n前 20 个 bad files:")

        for path, err in bad_summary_files[:20]:
            print(f"  - {path}")
            print(f"    Error: {err}")


def save_results_to_json(stats_by_category, save_path):
    output = {}

    for category, item in stats_by_category.items():
        num_tasks = item["num_tasks"]
        avg_success_rate = item["success_rate_sum"] / num_tasks if num_tasks > 0 else 0.0

        difficulty_output = {}

        for diff, diff_item in item["difficulty_levels"].items():
            diff_num_tasks = diff_item["num_tasks"]
            diff_avg_success_rate = (
                diff_item["success_rate_sum"] / diff_num_tasks
                if diff_num_tasks > 0
                else 0.0
            )

            difficulty_output[diff] = {
                "num_tasks": diff_num_tasks,
                "num_success": diff_item["num_success"],
                "avg_success_rate": diff_avg_success_rate,
            }

        output[category] = {
            "num_tasks": num_tasks,
            "num_success": item["num_success"],
            "avg_success_rate": avg_success_rate,
            "suites": sorted(list(item["suites"])),
            "match_methods": dict(item["match_methods"]),
            "difficulty_levels": difficulty_output,
            "tasks": item["tasks"],
        }

    total_tasks = sum(v["num_tasks"] for v in stats_by_category.values())
    total_success = sum(v["num_success"] for v in stats_by_category.values())
    total_success_rate_sum = sum(v["success_rate_sum"] for v in stats_by_category.values())

    output["overall"] = {
        "num_tasks": total_tasks,
        "num_success": total_success,
        "avg_success_rate": total_success_rate_sum / total_tasks if total_tasks > 0 else 0.0,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved result json to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result_root",
        type=str,
        default='/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/MagicBot-VGA/evaluation/Libero-plus/magicbotr0',
        help="包含 libero_goal/libero_spatial/libero_10/libero_object 结果目录的根路径。",
    )

    parser.add_argument(
        "--task_info_json",
        type=str,
        default='/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/LIBERO-plus/libero/libero/benchmark/task_classification.json',
        help="总任务描述 json，格式为 suite -> list of {id, name, category, difficulty_level}。",
    )

    parser.add_argument(
        "--suite_names",
        type=str,
        nargs="+",
        default=DEFAULT_SUITE_NAMES,
        help="需要统计的 suite 名称。",
    )

    parser.add_argument(
        "--difficulty",
        action="store_true",
        help="打印 category + difficulty_level 的分组成功率。",
    )

    parser.add_argument(
        "--detail",
        action="store_true",
        help="打印每个任务的详细结果。",
    )

    parser.add_argument(
        "--match_stat",
        default='True',
        help="打印 id_plus_1_match / name_prefix_fallback / unmatched 的数量。",
    )

    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="可选：保存统计结果到 json。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    stats_by_category, unmatched_tasks, bad_summary_files = analyze(
        result_root=args.result_root,
        task_info_json=args.task_info_json,
        suite_names=args.suite_names,
    )

    print_category_results(stats_by_category)

    if args.match_stat:
        print_match_method_results(stats_by_category)

    if args.difficulty:
        print_difficulty_results(stats_by_category)

    if args.detail:
        print_detail_results(stats_by_category)

    print_warnings(unmatched_tasks, bad_summary_files)

    if args.save_json is not None:
        save_results_to_json(stats_by_category, args.save_json)


if __name__ == "__main__":
    main()
