import unittest

from dalsa.answers import extract_mcq, is_correct, same_aime, same_math


class AnswerExtractionTests(unittest.TestCase):
    def test_mcq_ignores_lowercase_articles(self):
        text = "This is a simple explanation with a common article."
        self.assertIsNone(extract_mcq(text, labels="ABCD"))

    def test_mcq_extracts_explicit_final_answer(self):
        self.assertEqual(extract_mcq("Therefore, the final answer is b.", labels="ABCD"), "B")
        self.assertEqual(extract_mcq("Reasoning...\n\\boxed{C}", labels="ABCD"), "C")

    def test_math_fraction_matches_decimal(self):
        item = {"task_type": "math", "ground_truth": "0.5"}
        ok, pred = is_correct("Thus the answer is \\boxed{\\frac{1}{2}}.", item)
        self.assertTrue(ok)
        self.assertEqual(pred, "\\frac{1}{2}")

    def test_math_uses_previous_complete_box_when_later_box_is_truncated(self):
        item = {"task_type": "math", "ground_truth": "64"}
        ok, pred = is_correct("The final answer is \\boxed{64}. Repeating: \\boxed{", item)
        self.assertTrue(ok)
        self.assertEqual(pred, "64")

    def test_math_symbolic_does_not_fall_back_to_last_number(self):
        self.assertFalse(same_math("x^2 + 1", "1"))

    def test_explicit_math_expressions_are_evaluated_without_last_digit_fallback(self):
        for answer, gold, expected in [("1+2", "3", True), ("1+2", "2", False),
                                       ("2^3", "8", True), ("1/0", "0", False),
                                       (r"\sqrt{-1}", "1", False), ("sqrt", "0", False)]:
            with self.subTest(answer=answer, gold=gold):
                ok, _ = is_correct(r"\boxed{" + answer + "}", {"task_type": "math", "ground_truth": gold})
                self.assertEqual(ok, expected)

    def test_undefined_and_excessive_expressions_are_not_equal_to_themselves(self):
        for answer in ("1/0", "0/0", "inf", r"\infty", "nan", "oo", "zoo", "1/0, 2",
                       "2^1000000", "2^(2^20)", "2^1000^1000", "(2^1000)^1000"):
            with self.subTest(answer=answer):
                self.assertFalse(same_math(answer, answer))

    def test_safe_symbolic_functions_and_implicit_multiplication(self):
        for pred, gold in [("cos(x)", "cos(x)"), ("cos(0)", "1"), ("log(x)", "log(x)"),
                           ("a(x+1)", "a*x+a"), (r"\sin(0)", "0"), ("sin(pi/2)", "1")]:
            with self.subTest(pred=pred, gold=gold):
                self.assertTrue(same_math(pred, gold))
        self.assertFalse(same_math("sin(x)", "x"))
        self.assertTrue(same_math("[-2,7]", "x in [-2,7]"))

    def test_math_normalizes_units_degrees_and_assignments(self):
        self.assertTrue(same_math(r"42 \text{ inches}", "42"))
        self.assertTrue(same_math("90", r"90^\circ"))
        self.assertTrue(same_math("5", "x=5"))
        self.assertTrue(same_math(r"864 \mbox{ inches}^2", "864"))
        self.assertTrue(same_math(r"[-2, 7]", r"x \in [-2,7]"))

    def test_math_normalizes_lists_and_latex_spacing(self):
        self.assertTrue(same_math("-2, 1", "1,-2"))
        self.assertTrue(same_math("10080", r"10,\!080"))
        self.assertFalse(same_math("7", "3, 5, 7"))

    def test_math_normalizes_compact_fractions(self):
        self.assertTrue(same_math(r"\dfrac{4}{3}", r"\frac43"))
        self.assertTrue(same_math(r"\dfrac{9}{19}", r"\frac9{19}"))
        self.assertTrue(same_math("0.0000672", ".0000672"))

    def test_math_normalizes_choice_currency_and_symbolic_forms(self):
        self.assertTrue(same_math("C", r"\text{(C)}"))
        self.assertTrue(same_math("32348", r"\$32,\!348"))
        self.assertTrue(same_math(r"11(\sqrt{5}+1)", r"11\sqrt{5}+11"))
        self.assertTrue(same_math(r"\dfrac{9a + 11}{20}", r"\frac{11+9a}{20}"))

    def test_aime_rejects_huge_digit_strings_without_int_conversion(self):
        self.assertFalse(same_aime("1" * 5000, "123"))
        self.assertTrue(same_aime("007", "7"))

    def test_aime_requires_an_integer_in_the_answer_range(self):
        for answer, gold in [("-7", "7"), ("7.0", "70"), ("1/2", "12"), ("1000", "1000")]:
            with self.subTest(answer=answer):
                ok, _ = is_correct(r"\boxed{" + answer + "}", {"task_type": "aime", "ground_truth": gold})
                self.assertFalse(ok)
        self.assertTrue(same_aime("000", "0"))
        self.assertTrue(same_aime("999", "0999"))


if __name__ == "__main__":
    unittest.main()
