import contextvars
import logging
import unittest

from freebuff2api.logging_config import (
    ColorFormatter,
    ContextFilter,
    set_request_id,
    set_token_context,
)


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        "freebuff2api.test", logging.INFO, __file__, 1, "msg", (), None
    )


class LoggingConfigTests(unittest.TestCase):
    def test_color_formatter_adds_ansi_color(self) -> None:
        formatter = ColorFormatter("%(levelname)s %(message)s")
        record = logging.LogRecord(
            "freebuff2api.test",
            logging.INFO,
            __file__,
            1,
            "hello",
            (),
            None,
        )

        message = formatter.format(record)

        self.assertTrue(message.startswith("\033[32m"))
        self.assertTrue(message.endswith("\033[0m"))

    def test_context_filter_builds_prefix(self) -> None:
        def run() -> str:
            set_request_id("7f3a1c")
            set_token_context("t3/5", "U")
            record = _record()
            ContextFilter(color=False).filter(record)
            return record.ctx

        # run inside a fresh context so the vars don't leak across tests
        self.assertEqual(contextvars.copy_context().run(run), "[7f3a1c t3/5 U] ")

    def test_context_filter_colors_only_mode_letter(self) -> None:
        def run() -> str:
            set_request_id("7f3a1c")
            set_token_context("t1/5", "P")
            record = _record()
            ContextFilter(color=True).filter(record)
            return record.ctx

        ctx = contextvars.copy_context().run(run)
        self.assertIn("7f3a1c t1/5 ", ctx)
        self.assertIn("\033[1;33mP\033[0m\033[32m", ctx)

    def test_context_filter_empty_when_unset(self) -> None:
        def run() -> str:
            record = _record()
            ContextFilter(color=False).filter(record)
            return record.ctx

        self.assertEqual(contextvars.copy_context().run(run), "")


if __name__ == "__main__":
    unittest.main()
