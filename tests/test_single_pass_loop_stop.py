import unittest

from churro_decode_processors import RepetitiveXmlTailStoppingCriteria


class _Tokenizer:
    def decode(self, values, **kwargs):
        return "".join(chr(value) for value in values)


class SinglePassLoopStopTests(unittest.TestCase):
    def setUp(self):
        self.guard = RepetitiveXmlTailStoppingCriteria(_Tokenizer(), prompt_length=0)

    def test_detects_three_consecutive_repeated_line_blocks(self):
        self.assertEqual(
            self.guard._repeated_suffix(
                ["one two three four", "five six seven eight"] * 3
            ),
            (2, ["one two three four", "five six seven eight"]),
        )

    def test_does_not_stop_short_repeated_arithmetic_lines(self):
        self.assertIsNone(self.guard._repeated_suffix(["divide by 100"] * 3))

    def test_detects_three_consecutive_repeated_word_blocks(self):
        repeated = "the people are interested because they are tired " * 3
        width, block = self.guard._repeated_word_suffix(repeated)
        self.assertEqual(width, 8)
        self.assertEqual(block, "the people are interested because they are tired".split())

    def test_does_not_stop_nonconsecutive_repeated_words(self):
        value = "the people are interested today and the people were interested yesterday"
        self.assertIsNone(self.guard._repeated_word_suffix(value))

    def test_detects_growing_repetitive_prose_tail(self):
        value = (
            "The people are interested because they are not satisfied with the old "
            "established and accepted methods of religious instruction. They are not "
            "satisfied because they are tired of the old established and accepted "
            "methods of religious instruction. They are tired of the old established "
            "and accepted methods of religious instruction because they are tired of "
            "the old established and accepted methods of religious instruction."
        )
        audit = self.guard._low_novelty_word_tail(value)
        self.assertIsNotNone(audit)
        self.assertGreaterEqual(audit["duplicate_ngram_ratio"], 0.30)

    def test_clean_long_prose_does_not_trigger_low_novelty_guard(self):
        value = " ".join(
            f"sentence{index} contains distinct historical evidence number{index}"
            for index in range(20)
        )
        self.assertIsNone(self.guard._low_novelty_word_tail(value))

    def test_does_not_flag_nonconsecutive_recurring_lines(self):
        self.assertIsNone(
            self.guard._repeated_suffix(
                ["date", "first entry", "date", "second entry", "date", "third entry"]
            )
        )

    def test_normalization_ignores_case_and_punctuation(self):
        self.assertEqual(self.guard._normalize("  New York, DAY! "), "new york day")


if __name__ == "__main__":
    unittest.main()
