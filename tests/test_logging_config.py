import logging
import unittest

from freebuff2api.logging_config import ColorFormatter, mode_tag


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

    def test_mode_tag_plain_without_color(self) -> None:
        self.assertEqual(mode_tag("UNLIMITED", color=False), "[UNLIMITED]")
        self.assertEqual(mode_tag("PREMIUM", color=False), "[ PREMIUM ]")

    def test_mode_tag_colored_badge_restores_info_color(self) -> None:
        tag = mode_tag("PREMIUM", color=True)
        self.assertTrue(tag.startswith("\033[1;30;43m"))
        self.assertIn("[ PREMIUM ]", tag)
        # ends by restoring the INFO color so the rest of the line stays green
        self.assertTrue(tag.endswith("\033[0m\033[32m"))


if __name__ == "__main__":
    unittest.main()
