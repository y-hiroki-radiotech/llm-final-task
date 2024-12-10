from transformers import pipeline

def task_classification(dataset):
    model_name = "hiroki-rad/bert-base-classification-ft"
    classify_pipe = pipeline(model=model_name, device="cuda:0")

    results: list[dict[str, float | str]] = []
    for example in dataset.itertuples():
        # モデルの予測結果を取得
        model_prediction = classify_pipe(example.input)[0]
        # 正解のラベルIDをラベル名に変換
        results.append( model_prediction["label"])

    dataset["label"] = results

    return dataset
