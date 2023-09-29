import unittest

from . import mime as m


class StripSurroundingQuotesTestCase(unittest.TestCase):
    def test_no_quotes(self):
        self.assertEqual(m._strip_surrounding_quotes("foo"), "foo")

    def test_single_quotes(self):
        self.assertEqual(m._strip_surrounding_quotes("'foo'"), "foo")

    def test_double_quotes(self):
        self.assertEqual(m._strip_surrounding_quotes('"foo"'), "foo")

    def test_mismatched_quotes(self):
        self.assertEqual(m._strip_surrounding_quotes("'foo\""), "'foo\"")

    def test_missing_starting_quote(self):
        self.assertEqual(m._strip_surrounding_quotes("foo'"), "foo'")

    def test_missing_ending_quote(self):
        self.assertEqual(m._strip_surrounding_quotes('"foo'), '"foo')


class MimeTypeFromContentTypeTestCase(unittest.TestCase):
    def test_returns_basic_type(self):
        self.assertEqual(m.mime_type_from_content_type("text/html"), "text/html")

    def test_splits_content_type_extras(self):
        self.assertEqual(
            m.mime_type_from_content_type("text/html; charset=utf-8"), "text/html"
        )


class CharsetFromContentTypeTestCase(unittest.TestCase):
    def test_returns_charset(self):
        self.assertEqual(
            m.charset_from_content_type("text/html; charset=utf-8"), "utf-8"
        )

    def test_returns_none_if_no_charset(self):
        self.assertIsNone(m.charset_from_content_type("text/html"))

    def test_quoted_charset(self):
        self.assertEqual(
            m.charset_from_content_type('text/html; charset="utf-8"'), "utf-8"
        )

    def test_complex_content_type(self):
        self.assertEqual(
            m.charset_from_content_type(
                'text/html; foo=bar; charset="utf-8"; baz=quux'
            ),
            "utf-8",
        )


class SplitContentTypeTestCase(unittest.TestCase):
    def test_returns_mime_type_and_charset(self):
        self.assertEqual(
            m.split_content_type("text/html; charset=utf-8"),
            ("text/html", "utf-8"),
        )

    def test_returns_mime_type_and_none_if_no_charset(self):
        self.assertEqual(
            m.split_content_type("text/html"),
            ("text/html", None),
        )


class IsTextMimeTypeTestCase(unittest.TestCase):
    def test_returns_true_for_text_mime_types(self):
        self.assertTrue(m.is_text_mime_type("text/html"))

    def test_returns_false_for_non_text_mime_types(self):
        self.assertFalse(m.is_text_mime_type("image/png"))

    def test_returns_true_for_special_text_mime_types(self):
        self.assertTrue(m.is_text_mime_type("application/rss+xml"))


class IsTextContentTypeTestCase(unittest.TestCase):
    def test_returns_true_for_text_content_types(self):
        self.assertTrue(m.is_text_content_type("text/html; charset=utf-8"))

    def test_returns_false_for_non_text_content_types(self):
        self.assertFalse(m.is_text_content_type("image/png"))

    def test_returns_true_for_special_text_content_types(self):
        self.assertTrue(m.is_text_content_type("application/rss+xml; charset=utf-8"))


class IsBinaryMimeTypeTestCase(unittest.TestCase):
    def test_returns_true_for_binary_mime_types(self):
        self.assertTrue(m.is_binary_mime_type("image/png"))

    def test_returns_false_for_non_binary_mime_types(self):
        self.assertFalse(m.is_binary_mime_type("text/html"))

    def test_returns_false_for_special_text_mime_types(self):
        self.assertFalse(m.is_binary_mime_type("application/rss+xml"))


class IsBinaryContentTypeTestCase(unittest.TestCase):
    def test_returns_true_for_binary_content_types(self):
        self.assertTrue(m.is_binary_content_type("image/png"))

    def test_returns_false_for_non_binary_content_types(self):
        self.assertFalse(m.is_binary_content_type("text/html; charset=utf-8"))

    def test_returns_false_for_special_text_content_types(self):
        self.assertFalse(m.is_binary_content_type("application/rss+xml; charset=utf-8"))
