import pandas as pd

# JSONLファイルを読み込む
file_path = 'elyza-tasks-100-TV_0.jsonl'
dataset = pd.read_json(file_path, lines=True)
