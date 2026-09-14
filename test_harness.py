"""python -m unittest test_harness -v

No network, no keys, no dependencies. The HTTP client tests run against a throwaway
local server so request shaping and error handling are genuinely exercised.
"""
import os
import shutil
import tempfile
import threading
import unittest
from collections import Counter, defaultdict
from types import SimpleNamespace

from harness import (STRATEGIES, MockClient, ResponseCache, parse_cot, parse_json,
                     parse_plain)
from metrics import (discordant_pairs, macro_f1, mcnemar_exact, mean_and_sd, per_class_prf,
                     percentile, wilson_ci)
from evaluate import take_sample
from providers import OpenAICompatibleClient

os.environ["PROMPTBENCH_QUIET"] = "1"  # keep operational notices out of test output


class TestPlainParser(unittest.TestCase):
    def test_bare_label(self):
        self.assertEqual(parse_plain("billing"), ("billing", "ok"))

    def test_case_and_punctuation(self):
        self.assertEqual(parse_plain("  Technical.  "), ("technical", "ok"))

    def test_strips_prefix(self):
        self.assertEqual(parse_plain("Category: account"), ("account", "ok"))

    def test_takes_last_line_past_filler(self):
        self.assertEqual(parse_plain("Sure, happy to help.\nother"), ("other", "ok"))

    def test_empty_is_empty_not_malformed(self):
        self.assertEqual(parse_plain("   \n "), (None, "empty"))

    def test_short_unknown_label_is_hallucinated(self):
        self.assertEqual(parse_plain("billing-and-payments")[1], "hallucinated")

    def test_long_prose_is_malformed(self):
        self.assertEqual(parse_plain("I am not sure about this one.")[1], "malformed")


class TestCotParser(unittest.TestCase):
    def test_marker(self):
        self.assertEqual(parse_cot("Reasoning here.\nANSWER: technical"), ("technical", "ok"))

    def test_case_insensitive_marker(self):
        self.assertEqual(parse_cot("answer:  Billing"), ("billing", "ok"))

    def test_takes_last_marker_when_model_restates(self):
        self.assertEqual(parse_cot("ANSWER: billing\nOn reflection.\nANSWER: account"),
                         ("account", "ok"))

    def test_missing_marker_is_malformed(self):
        self.assertEqual(parse_cot("I think this is a billing issue.")[1], "malformed")

    def test_invalid_category_inside_valid_format(self):
        self.assertEqual(parse_cot("ANSWER: billing-and-payments")[1], "hallucinated")


class TestJsonParser(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(parse_json('{"category": "account"}'), ("account", "ok"))

    def test_fenced(self):
        self.assertEqual(parse_json('```json\n{"category": "other"}\n```'), ("other", "ok"))

    def test_extra_keys_are_fine(self):
        self.assertEqual(parse_json('{"category": "billing", "confidence": 0.9}'),
                         ("billing", "ok"))

    def test_broken_json_is_malformed(self):
        self.assertEqual(parse_json('{"category": ')[1], "malformed")

    def test_missing_key_is_malformed(self):
        self.assertEqual(parse_json('{"label": "billing"}')[1], "malformed")

    def test_non_object_is_malformed(self):
        self.assertEqual(parse_json('["billing"]')[1], "malformed")

    def test_valid_json_invalid_category(self):
        self.assertEqual(parse_json('{"category": "refunds"}')[1], "hallucinated")


class TestMetrics(unittest.TestCase):
    def test_wilson_stays_in_bounds_at_perfect_score(self):
        lo, hi = wilson_ci(40, 40)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 1.0)  # 40/40 is not proof of perfection

    def test_wilson_is_wide_at_n_40(self):
        lo, hi = wilson_ci(34, 40)
        self.assertGreater(hi - lo, 0.15)

    def test_wilson_zero_n(self):
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_mcnemar_symmetric_is_p_one(self):
        self.assertEqual(mcnemar_exact(5, 5), 1.0)

    def test_mcnemar_no_discordant_pairs(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)

    def test_mcnemar_lopsided_is_significant(self):
        self.assertLess(mcnemar_exact(10, 0), 0.01)

    def test_two_discordant_items_can_never_be_evidence(self):
        self.assertGreater(mcnemar_exact(2, 0), 0.05)

    def test_discordant_pairs(self):
        self.assertEqual(discordant_pairs([1, 1, 0, 0], [1, 0, 1, 0]), (1, 1))

    def test_discordant_pairs_rejects_misalignment(self):
        with self.assertRaises(ValueError):
            discordant_pairs([1, 0], [1, 0, 1])

    def test_percentile_nearest_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.95), 5)
        self.assertEqual(percentile([], 0.95), 0.0)

    def test_mean_and_sd_single_value(self):
        self.assertEqual(mean_and_sd([0.8]), (0.8, 0.0))

    def test_invalid_column_hurts_recall_not_precision(self):
        conf = defaultdict(Counter)
        conf["billing"]["billing"] = 8
        conf["billing"]["INVALID"] = 2
        conf["technical"]["technical"] = 10
        conf["account"]["account"] = 10
        conf["other"]["other"] = 10
        prf = per_class_prf(conf, ["billing", "technical", "account", "other"])
        self.assertAlmostEqual(prf["billing"]["recall"], 0.8)
        self.assertAlmostEqual(prf["billing"]["precision"], 1.0)
        self.assertLess(macro_f1(prf), 1.0)


class TestStrategyRegistry(unittest.TestCase):
    def test_every_strategy_has_a_builder_and_parser(self):
        for name, s in STRATEGIES.items():
            prompt = s.build("my invoice is wrong")
            self.assertIn("invoice", prompt, name)
            self.assertGreater(s.max_tokens, 0, name)

    def test_only_few_shot_needs_stop_sequences(self):
        # The completion-style prompt is the one that runs on past its answer.
        self.assertTrue(STRATEGIES["few_shot"].stop)
        self.assertFalse(STRATEGIES["few_shot_instruct"].stop)

    def test_instruct_variant_keeps_the_examples(self):
        prompt = STRATEGIES["few_shot_instruct"].build("x")
        self.assertIn("freezes on the login screen", prompt)
        self.assertFalse(prompt.rstrip().endswith("Category:"))


class TestSampling(unittest.TestCase):
    def _rows(self):
        rows = []
        for i, lab in enumerate(["billing", "technical", "account", "other"]):
            for j in range(10):
                rows.append({"id": str(i * 10 + j + 1), "message": f"m{i}{j}", "label": lab})
        return rows

    def test_limit_spreads_across_classes(self):
        # dataset.csv is grouped by label, so a plain slice would return one class.
        picked = take_sample(self._rows(), 8)
        self.assertEqual(len(picked), 8)
        self.assertEqual(Counter(r["label"] for r in picked),
                         Counter({"billing": 2, "technical": 2, "account": 2, "other": 2}))

    def test_limit_is_deterministic(self):
        self.assertEqual(take_sample(self._rows(), 8), take_sample(self._rows(), 8))

    def test_limit_larger_than_dataset_returns_everything(self):
        rows = self._rows()
        self.assertEqual(take_sample(rows, 999), rows)
        self.assertEqual(take_sample(rows, 0), rows)

    def test_uneven_limit_still_fills(self):
        picked = take_sample(self._rows(), 7)
        self.assertEqual(len(picked), 7)


class TestMockDeterminism(unittest.TestCase):
    def test_reset_reproduces_the_same_schedule(self):
        m = MockClient()
        first = [m.complete("", "", "zero_shot", "my invoice is wrong", 16).text
                 for _ in range(20)]
        m.reset()
        second = [m.complete("", "", "zero_shot", "my invoice is wrong", 16).text
                  for _ in range(20)]
        self.assertEqual(first, second)

    def test_different_runs_give_different_schedules(self):
        m = MockClient()
        m.reset(0)
        a = [m.complete("", "", "zero_shot", "my invoice is wrong", 16).text for _ in range(30)]
        m.reset(1)
        b = [m.complete("", "", "zero_shot", "my invoice is wrong", 16).text for _ in range(30)]
        self.assertNotEqual(a, b)  # otherwise --repeats would report sd=0 forever

    def test_all_four_failure_modes_are_reachable(self):
        m = MockClient()
        seen = set()
        for _ in range(200):
            text = m.complete("", "", "strict_json", "my invoice is wrong", 64).text
            label, status = parse_json(text)
            seen.add("correct" if label == "billing" and status == "ok" else status)
        self.assertEqual(seen, {"correct", "ok", "hallucinated", "malformed", "empty"})


class TestOpenAICompatibleClient(unittest.TestCase):
    """Runs against a throwaway local server rather than a mocked urlopen, so request
    shaping, auth headers and response parsing are all actually exercised."""

    LAST = []

    @classmethod
    def setUpClass(cls):
        import json
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self._send(200, {"data": [{"id": "model-a"}, {"id": "model-b"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                TestOpenAICompatibleClient.LAST.append(body)
                if self.headers.get("Authorization") != "Bearer test-token":
                    return self._send(401, {"error": {"message": "Bad credentials"}})
                if body.get("model") == "boom":
                    return self._send(500, {"error": {"message": "server on fire"}})
                if body.get("model") == "picky" and "max_tokens" in body:
                    return self._send(400, {"error": {"message":
                                                      "use max_completion_tokens instead"}})
                self._send(200, {"choices": [{"message": {"content": "billing"}}],
                                 "usage": {"prompt_tokens": 90, "completion_tokens": 4}})

            def _send(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        cls.server = HTTPServer(("127.0.0.1", 0), Handler)  # 0 picks a free port
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/v1"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def _client(self, model="model-a", key="test-token", **kw):
        os.environ["PROMPTBENCH_TEST_KEY"] = key
        return OpenAICompatibleClient(self.base, model, "PROMPTBENCH_TEST_KEY", **kw)

    def last_payload(self):
        return type(self).LAST[-1]

    def test_successful_call_parses_text_and_usage(self):
        out = self._client().complete("sys", "user", "zero_shot", "msg", 16)
        self.assertEqual(out.text, "billing")
        self.assertEqual((out.input_tokens, out.output_tokens), (90, 4))
        self.assertEqual(out.error, "")
        self.assertGreater(out.latency_ms, 0)

    def test_bad_key_is_not_retried(self):
        out = self._client(key="wrong", max_retries=4).complete("s", "u", "zero_shot", "m", 16)
        self.assertIn("401", out.error)
        self.assertEqual(out.attempts, 1)

    def test_server_error_is_retried_then_reported(self):
        out = self._client(model="boom", max_retries=2).complete("s", "u", "zero_shot", "m", 16)
        self.assertIn("500", out.error)
        self.assertEqual(out.attempts, 2)

    def test_switches_to_max_completion_tokens_when_told_to(self):
        client = self._client(model="picky")
        out = client.complete("s", "u", "zero_shot", "m", 16)
        self.assertEqual(out.text, "billing")
        self.assertEqual(client._token_field, "max_completion_tokens")

    def test_stop_sequences_are_sent(self):
        client = self._client(model="echo")
        client.complete("s", "u", "few_shot", "m", 32, None, ("\nMessage:", "\n\n"))
        self.assertEqual(self.last_payload().get("stop"), ["\nMessage:", "\n\n"])

    def test_no_stop_key_when_strategy_has_none(self):
        client = self._client(model="echo")
        client.complete("s", "u", "zero_shot", "m", 16)
        self.assertNotIn("stop", self.last_payload())

    def test_cache_round_trip_skips_the_second_call(self):
        tmpdir = tempfile.mkdtemp()
        try:
            cache = ResponseCache(tmpdir)
            client = self._client(cache=cache)
            before = len(type(self).LAST)
            key = ResponseCache.key("model-a", 0.0, "zero_shot", 16, (), 0, "prompt")
            first = client.complete("sys", "prompt", "zero_shot", "msg", 16, key)
            second = client.complete("sys", "prompt", "zero_shot", "msg", 16, key)
            self.assertEqual(len(type(self).LAST) - before, 1)  # second never hit the server
            self.assertFalse(first.cached)
            self.assertTrue(second.cached)
            self.assertEqual(second.text, first.text)
            self.assertEqual(second.input_tokens, 90)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_cache_key_separates_runs(self):
        # Without this, --repeats would replay one cached answer and report sd=0 forever.
        a = ResponseCache.key("m", 0.0, "zero_shot", 16, (), 0, "prompt")
        b = ResponseCache.key("m", 0.0, "zero_shot", 16, (), 1, "prompt")
        self.assertNotEqual(a, b)

    def test_lists_models(self):
        self.assertEqual(self._client().list_models(), ["model-a", "model-b"])

    def test_missing_key_exits_with_guidance(self):
        os.environ.pop("PROMPTBENCH_ABSENT_KEY", None)
        with self.assertRaises(SystemExit):
            OpenAICompatibleClient(self.base, "model-a", "PROMPTBENCH_ABSENT_KEY")

    def test_key_optional_provider_needs_no_key(self):
        os.environ.pop("PROMPTBENCH_ABSENT_KEY", None)
        client = OpenAICompatibleClient(self.base, "model-a", "PROMPTBENCH_ABSENT_KEY",
                                        key_optional=True)
        self.assertEqual(client.key, "")


if __name__ == "__main__":
    unittest.main()
