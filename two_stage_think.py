import time
from vllm import LLM, SamplingParams

class TwoStageThinking:
    def __init__(self, llm):
        """
        二段階思考テキスト生成器の初期化

        Args:
            llm: 使用する言語モデル
        """
        self.llm = llm
        self.sampling_params = SamplingParams(
            repetition_penalty=1.2,
            temperature=0.3,
            max_tokens=2048
        )

    def _generate_text(self, prompt):
        """
        テキスト生成の共通処理

        Args:
            prompt (str): 生成用のプロンプト

        Returns:
            str: 生成されたテキスト
        """
        try:
            outputs = self.llm.generate(prompt, self.sampling_params)
            return outputs[0].outputs[0].text.strip()

        except Exception as e:
            print(f"テキスト生成エラー: {str(e)}")
            return ""

    def first_thinking(self, data, few_shot_example):
        """
        一段階目の思考生成

        Args:
            data: 入力データ（.inputフィールドを持つオブジェクト）

        Returns:
            str: 生成された回答
        """
        prompt = f"""## 指示:あなたは優秀な日本人の問題解決のエキスパートです。以下のステップで質問に取り組んでください：\n\n1. 質問の種類を特定する（事実確認/推論/創造的回答/計算など）\n2. 重要な情報や制約条件を抽出する\n3. 解決に必要なステップを明確にする\n4. 回答を組み立てる\n\n
        回答例は以下のものを参考にできます: {few_shot_example}\n\n
        質問をよく読んで、じっくり冷静に考え、考えをステップバイステップでまとめましょう。
        回答を作成する前に深呼吸して、分析結果を踏まえて解決しているかよく考えましょう。
        適切な回答を簡潔に出力してください。

        質問:{data.input}\n回答: """


        return self._generate_text(prompt)

    def second_thinking(self, data, first_output):
        """
        二段階目の思考生成

        Args:
            data: 入力データ（.inputフィールドを持つオブジェクト）
            first_output (str): 一段階目の出力

        Returns:
            str: 生成された回答
        """
        prompt = f"""## 指示:これは一度あなたが考えた回答です。\n#あなたの回答:{first_output}

        日本語で簡潔に答えられているかチェックして、必要であれば修正してください。
        回答:　"""

        return self._generate_text(prompt)

    def generate_complete_response(self, data, few_shot_example):
        """
        完全な二段階思考プロセスを実行

        Args:
            data: 入力データ（.inputフィールドを持つオブジェクト）

        Returns:
            tuple: (一段階目の回答, 二段階目の回答)
        """
        first_result = self.first_thinking(data,  few_shot_example.split("->")[1])
        time.sleep(10)
        return first_result
