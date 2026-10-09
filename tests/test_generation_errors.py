"""Generation diagnostics: offline, synthetic responses, no Cocoa or real API calls."""
import ast
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import generate
import settings_config
import styles

CREDS = ("https://example.invalid/v1", "synthetic-key", "test-model", "test", "openai")
PRIVATE = "PRIVATE-RESPONSE-KEY-CHAT-CONTENT"


def hud_harness():
    tree = ast.parse((ROOT / "src/hud.py").read_text())
    source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HudController")
    names = {"_payload_from_gen", "_finish_generate", "_regen_work", "_regenerate_work",
             "_push", "_push_reply", "applyReplyUpdate_"}
    methods = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for method in methods:
        method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name="Harness", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[]))
    log = Mock()
    scope = {"time": time, "_log": log,
             "chat_context": SimpleNamespace(model_message=lambda message, context: message)}
    exec(compile(module, str(ROOT / "src/hud.py"), "exec"), scope)
    h = scope["Harness"]()
    h._reply_current = lambda: True
    h.reload_conversations = Mock()
    h._paused = False
    h._app = object()
    h._reply_epoch = 1
    h._reply_worker = SimpleNamespace(context=None, epoch=1)
    h._active_context = None
    h._stream_hook = Mock(return_value=None)
    h._rank_payload = Mock(side_effect=lambda payload, *args: payload)
    h.generator = Mock()
    h.applyError_ = Mock()
    h.performSelectorOnMainThread_withObject_waitUntilDone_ = Mock()
    return h, log


def run_hud(h, path, result):
    h.generator.generate.return_value = result
    if path == "initial":
        h._finish_generate(result, SimpleNamespace(text="synthetic"), time.perf_counter(), None)
    elif path == "tone":
        h._regen_work("synthetic", "", [next(iter(styles.PRESETS))])
    else:
        h._regenerate_work("synthetic", "", [next(iter(styles.PRESETS))])


class GenerationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(generate, "load_credentials", return_value=CREDS))
        self.enterContext(patch.object(generate, "_extra_params", return_value={}))
        self.g = generate.Generator()
        self.tones = [next(iter(styles.PRESETS))]

    def result_for(self, error):
        with patch.object(self.g, "_call", side_effect=error):
            return self.g.generate("synthetic", slot_tones=self.tones)

    def test_known_errors_survive_all_hud_paths_without_raw_details(self):
        errors = [
            (urllib.error.HTTPError("https://example.invalid/" + PRIVATE, 401, PRIVATE, {}, io.BytesIO(PRIVATE.encode())), "HTTP 401"),
            (urllib.error.HTTPError("https://example.invalid/", 429, PRIVATE, {}, None), "HTTP 429"),
            (generate.ThinkingOnlyError(PRIVATE), "模型仅返回思考内容"),
            (generate.OutputLimitError(PRIVATE), "输出达到长度上限但没有正文"),
            (TimeoutError(PRIVATE), "TimeoutError"),
        ]
        for error, expected in errors:
            result = self.result_for(error)
            self.assertIn(expected, result["error"])
            self.assertNotIn(PRIVATE, repr(result))
            for path in ("initial", "tone", "regenerate"):
                with self.subTest(error=expected, path=path):
                    h, log = hud_harness()
                    run_hud(h, path, result)
                    args = h.performSelectorOnMainThread_withObject_waitUntilDone_.call_args.args
                    h.applyReplyUpdate_(args[1])
                    self.assertIn(expected, h.applyError_.call_args.args[0])
                    self.assertNotIn(PRIVATE, repr(log.mock_calls))
                    self.assertNotIn(PRIVATE, repr(h.applyError_.mock_calls))

    def test_defensive_worker_exception_does_not_include_message(self):
        with patch.object(self.g, "_one_tone", side_effect=ValueError(PRIVATE)):
            result = self.g.generate("synthetic", slot_tones=self.tones)
        self.assertEqual(result["error"], "ValueError")
        self.assertNotIn(PRIVATE, repr(result))

    def test_empty_response_keeps_safe_fallback_on_all_paths(self):
        with patch.object(self.g, "_call", return_value=""):
            result = self.g.generate("synthetic", slot_tones=self.tones)
        self.assertEqual(result["error"], "")
        for path in ("initial", "tone", "regenerate"):
            with self.subTest(path=path):
                h, _ = hud_harness()
                run_hud(h, path, result)
                h.applyReplyUpdate_(h.performSelectorOnMainThread_withObject_waitUntilDone_.call_args.args[1])
                self.assertIn("服务未返回可用候选，请检查模型设置", h.applyError_.call_args.args[0])

    def test_stale_errors_stay_guarded_on_all_paths(self):
        result = self.result_for(TimeoutError(PRIVATE))
        for path in ("initial", "tone", "regenerate"):
            with self.subTest(path=path):
                h, _ = hud_harness()
                run_hud(h, path, result)
                update = h.performSelectorOnMainThread_withObject_waitUntilDone_.call_args.args[1]
                h._reply_epoch += 1
                h.applyReplyUpdate_(update)
                h.applyError_.assert_not_called()

    def test_missing_key_does_not_expose_credentials_or_call_provider(self):
        with patch.object(generate, "load_credentials", return_value=("", "", "", "none", "openai")), \
             patch.object(self.g, "_call", side_effect=AssertionError("no request")) as call:
            result = self.g.generate("synthetic", slot_tones=self.tones)
        self.assertEqual(result["error"], generate.MISSING_HINT)
        call.assert_not_called()

    def test_partial_success_is_still_usable(self):
        tones = list(styles.PRESETS)[:2]
        def one_tone(message, intent, tone, *args):
            return (["usable"], "") if tone == tones[0] else ([], "HTTP 429：请检查模型服务设置")
        with patch.object(self.g, "_one_tone", side_effect=one_tone):
            result = self.g.generate("synthetic", slot_tones=tones)
        self.assertEqual(result["error"], "")
        h, _ = hud_harness()
        self.assertEqual(h._payload_from_gen(result)[0][2][0]["text"], "usable")

    def test_openai_no_body_length_is_distinct_from_empty(self):
        for content in (None, "", "   "):
            with self.subTest(content=content), self.assertRaises(generate.OutputLimitError):
                self.g._openai_json({"choices": [{"message": {"content": content}, "finish_reason": "length"}]}, "model", "alt")
        self.assertEqual(self.g._openai_json({}, "model", "alt"), "")
        self.assertEqual(self.g._openai_json({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}, "model", "alt"), "")

    def test_visible_reasoning_keeps_thinking_diagnosis(self):
        for field in ("reasoning_content", "reasoning"):
            with self.subTest(field=field), self.assertRaises(generate.ThinkingOnlyError):
                self.g._openai_json({"choices": [{"message": {"content": "", field: PRIVATE}, "finish_reason": "length"}]}, "model", "alt")

    def test_length_does_not_discard_nonempty_text(self):
        self.assertEqual(self.g._openai_json({"choices": [{"message": {"content": "usable"}, "finish_reason": "length"}]}, "model", "alt"), "usable")

    def stream(self, events):
        payload = b"".join(("data: " + json.dumps(evt) + "\n\n").encode() for evt in events)
        response = io.BytesIO(payload + b"data: [DONE]\n\n")
        response.headers = {"content-type": "text/event-stream"}
        with patch.object(generate.urllib.request, "urlopen", return_value=response):
            return self.g._stream_openai("https://example.invalid/v1/chat/completions", {}, {}, "model", "alt", lambda _: None)

    def test_stream_length_without_text(self):
        with self.assertRaises(generate.OutputLimitError):
            self.stream([{"choices": [{"delta": {"content": ""}}]}, {"choices": [{"delta": {}, "finish_reason": "length"}]}])

    def test_stream_reasoning_remains_private_and_takes_priority(self):
        with self.assertRaises(generate.ThinkingOnlyError) as caught:
            self.stream([{"choices": [{"delta": {"reasoning_content": PRIVATE}}]}, {"choices": [{"delta": {}, "finish_reason": "length"}]}])
        self.assertNotIn(PRIVATE, str(caught.exception))

    def test_stream_keeps_nonempty_text_and_plain_empty(self):
        self.assertEqual(self.stream([{"choices": [{"delta": {"content": "usable"}, "finish_reason": "length"}]}]), "usable")
        self.assertEqual(self.stream([{"choices": [{"delta": {}, "finish_reason": "stop"}]}]), "")

    def test_anthropic_empty_length_and_thinking(self):
        cases = [({"content": [], "stop_reason": "max_tokens"}, generate.OutputLimitError),
                 ({"content": [{"type": "thinking", "thinking": PRIVATE}], "stop_reason": "max_tokens"}, generate.ThinkingOnlyError)]
        for data, expected in cases:
            with self.subTest(error=expected.__name__), \
                 patch.object(generate, "load_credentials", return_value=(*CREDS[:-1], "anthropic")), \
                 patch.object(self.g, "_post", return_value=data), self.assertRaises(expected):
                self.g._call("synthetic")

    def test_settings_output_limit_error_is_safe(self):
        self.assertEqual(settings_config.error_message(generate.OutputLimitError(PRIVATE)), generate.OUTPUT_LIMIT_HINT)

    def test_extra_body_thinking_opt_out_is_preserved(self):
        extra = {"thinking": {"type": "disabled"}}
        with patch.object(generate, "_extra_params", return_value=extra), \
             patch.object(self.g, "_post", return_value={"choices": [{"message": {"content": "usable"}}]}) as post:
            self.g._call("synthetic")
        body = post.call_args.args[2]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(body["max_tokens"], 300)


if __name__ == "__main__":
    unittest.main()
