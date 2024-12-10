from transformers import pipeline

def task_classification(dataset):
    """
    指定されたデータセットに対してタスク分類を行う関数。

    この関数は、指定されたデータセットの各入力に対してBERTモデルを使用して分類を行い、
    予測されたラベルをデータセットに追加します。

    Args:
        dataset (pd.DataFrame): 'input'列を持つデータフレーム。各行は分類対象のテキストデータを含む。

    Returns:
        pd.DataFrame: 予測されたラベルが追加されたデータフレーム。
    """
    model_name = "hiroki-rad/bert-base-classification-ft"
    classify_pipe = pipeline(model=model_name, device="cuda:0")

    results: list[dict[str, float | str]] = []
    for example in dataset.itertuples():
        # モデルの予測結果を取得
        model_prediction = classify_pipe(example.input)[0]
        # 正解のラベルIDをラベル名に変換
        results.append(model_prediction["label"])

    dataset["label"] = results

    return dataset
