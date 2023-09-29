def _strip_surrounding_quotes(s: str) -> str:
    """Remove surrounding quotes -- single or double -- from a string."""
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    else:
        return s


def mime_type_from_content_type(content_type: str) -> str:
    """
    Return the MIME type from a Content-Type header value.
    """
    return content_type.split(";")[0].strip().lower()


def charset_from_content_type(content_type: str) -> str | None:
    """
    Return the charset from a Content-Type header value, if appropriate.
    """
    for part in content_type.split(";"):
        if part.strip().lower().startswith("charset="):
            charset = part.strip().split("=")[1].strip().lower()
            return _strip_surrounding_quotes(charset)

    return None


def split_content_type(content_type: str) -> tuple[str, str | None]:
    """
    Break a Content-Type header value into its MIME type and charset components.
    """
    mime_type = mime_type_from_content_type(content_type)
    charset = charset_from_content_type(content_type)
    return mime_type, charset


# Collection of mime types that are typically treated as text but do
# NOT start with "text/". Is there a principled way to do this?
TEXT_MIME_TYPES = {
    # JSON
    "application/json",
    # XML
    "application/xml",
    # RSS
    "application/rss+xml",
    # Atom
    "application/atom+xml",
    # JSON Feed
    "application/feed+json",
    # RDF XML
    "application/rdf+xml",
    # Rich text
    "application/rtf",
    # SVG XML
    "image/svg+xml",
    # XHTML
    "application/xhtml+xml",
}


def is_text_mime_type(mime_type: str) -> bool:
    """Return True if the MIME type is text."""
    return mime_type.startswith("text/") or mime_type in TEXT_MIME_TYPES


def is_text_content_type(content_type: str) -> bool:
    """Return True if the Content-Type is text."""
    mime_type, _ = split_content_type(content_type)
    return is_text_mime_type(mime_type)


def is_binary_mime_type(mime_type: str) -> bool:
    """Return True if the MIME type is binary."""
    return not is_text_mime_type(mime_type)


def is_binary_content_type(content_type: str) -> bool:
    """Return True if the Content-Type is binary."""
    mime_type, _ = split_content_type(content_type)
    return is_binary_mime_type(mime_type)
