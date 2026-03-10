import json, collections, pathlib

# 1. 读文件并分组
groups = collections.defaultdict(list)
for line in pathlib.Path('train_dataset.filtered.jsonl').read_text(encoding='utf8').splitlines():
    if line.strip():
        d = json.loads(line)
        groups[d['UserId']].append(d)

# 2. 按回答数量降序排序，取前 20
top20 = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)[:20]

# 3. 打印结果
for rank, (uid, records) in enumerate(top20, 1):
    print(f'No.{rank:3d}  UserId:{uid}  回答数:{len(records)}')

# 4. 合并保存
with open('top20_users_records.jsonl', 'w', encoding='utf8') as f_out:
    for uid, records in top20:
        for rec in records:
            json.dump(rec, f_out, ensure_ascii=False)
            f_out.write('\n')


            