import json
import tempfile
import unittest
from pathlib import Path

from tepid_h1.evaluation import (
    generate_retrieval_suite,
    load_answer_key,
    load_predictions,
    score_retrieval,
    write_retrieval_suite,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_generator_is_deterministic_and_exact_length(self):
        first = generate_retrieval_suite(lengths=(64, 128), seed=7)
        second = generate_retrieval_suite(lengths=(64, 128), seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        for case in first:
            self.assertEqual(len(case.prompt.split()), case.target_tokens)
            self.assertIn(case.expected_answer, case.prompt)
            self.assertGreater(case.insertion_index, 0)

    def test_prompt_and_answer_files_are_separate(self):
        cases = generate_retrieval_suite(lengths=(64,), positions=(0.5,), seed=9)
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.jsonl"
            answers = Path(directory) / "answers.jsonl"
            write_retrieval_suite(cases, prompts_path=prompts, answers_path=answers)

            prompt_record = json.loads(prompts.read_text().strip())
            answer_record = json.loads(answers.read_text().strip())

        self.assertNotIn("answer", prompt_record)
        self.assertEqual(answer_record["answer"], cases[0].expected_answer)

    def test_oracle_predictions_pass_all_breakdowns(self):
        cases = generate_retrieval_suite(lengths=(64, 128), seed=11)
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.jsonl"
            answers_path = Path(directory) / "answers.jsonl"
            write_retrieval_suite(cases, prompts_path=prompts, answers_path=answers_path)
            answers = load_answer_key(answers_path)
            predictions = load_predictions(answers_path)

        report = score_retrieval(answers, predictions)

        self.assertTrue(report["passed"])
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(set(report["by_length"]), {"64", "128"})
        self.assertTrue(all(item["accuracy"] == 1.0 for item in report["by_position"].values()))

    def test_missing_and_incorrect_predictions_fail(self):
        cases = generate_retrieval_suite(lengths=(64,), positions=(0.1, 0.9), seed=13)
        answers = {case.case_id: case.answer_record() for case in cases}
        predictions = {cases[0].case_id: "wrong"}

        report = score_retrieval(answers, predictions, minimum_accuracy=0.0)

        self.assertFalse(report["passed"])
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual({item["reason"] for item in report["failures"]}, {"missing", "incorrect"})
        self.assertEqual(
            report["by_position"]["middle"], {"cases": 0, "correct": 0, "accuracy": None}
        )

    def test_generate_retrieval_suite_requires_valid_lengths(self):
        with self.assertRaisesRegex(ValueError, "at least 32"):
            generate_retrieval_suite(lengths=(16,), seed=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            generate_retrieval_suite(lengths=(64, 64), seed=1)

    def test_generate_retrieval_suite_requires_valid_positions(self):
        with self.assertRaisesRegex(ValueError, "in \\(0, 1\\)"):
            generate_retrieval_suite(lengths=(64,), positions=(0.0,), seed=1)
        with self.assertRaisesRegex(ValueError, "unique"):
            generate_retrieval_suite(lengths=(64,), positions=(0.5, 0.5), seed=1)

    def test_write_and_load_retrieval_suite(self):
        cases = generate_retrieval_suite(lengths=(64,), positions=(0.5,), seed=42)
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.jsonl"
            answers = Path(directory) / "answers.jsonl"
            write_retrieval_suite(cases, prompts_path=prompts, answers_path=answers)

            loaded_answers = load_answer_key(answers)

            self.assertEqual(len(loaded_answers), 1)
            self.assertIn("answer", loaded_answers[cases[0].case_id])


if __name__ == "__main__":
    unittest.main()
