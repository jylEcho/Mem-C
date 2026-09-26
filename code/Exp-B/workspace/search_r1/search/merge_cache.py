import json
import argparse
import sys

def merge_json_files(file1_path, file2_path, output_path):
    print(f"[*] start merging...\n    file1: {file1_path}\n    file2: {file2_path}")

    # 读取第一个文件
    try:
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        if not isinstance(data1, dict):
            print(f"[!] error: file '{file1_path}' 的内容不是一个有效的JSON对象 (字典)。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"[!] error: cannot find file '{file1_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[!] error: file '{file1_path}' 不是有效的JSON格式。")
        sys.exit(1)

    # 读取第二个文件
    try:
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
        if not isinstance(data2, dict):
            print(f"[!] error: file '{file2_path}' 的内容不是一个有效的JSON对象 (字典)。")
            sys.exit(1)
    except FileNotFoundError:
        print(f"[!] error: cannot find file '{file2_path}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[!] error: file '{file2_path}' 不是有效的JSON格式。")
        sys.exit(1)

    # 合并字典
    print("[*] 正在合并数据...")
    merged_cache = data1.copy()  # 从第一个字典的副本开始，以免修改原始数据
    merged_cache.update(data2)   # 将第二个字典的数据更新进来
    
    # 保存合并后的文件 (使用您指定的逻辑)
    print(f"[*] 正在将合并结果保存到: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_cache, f, ensure_ascii=False, indent=2)
        print(f"[+] 成功！合并后的文件已保存为 '{output_path}'")
    except IOError as e:
        print(f"[!] 错误: 无法写入文件 '{output_path}'. 错误信息: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="merge to json cache")
    parser.add_argument("file1", help="first file")
    parser.add_argument("file2", help="second file")
    parser.add_argument("output", help="output file")

    # parse args
    args = parser.parse_args()

    # merge
    merge_json_files(args.file1, args.file2, args.output)