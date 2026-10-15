SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text, chunk_size, separators):
    """Cut text into pieces that are each at most chunk_size long."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:
        # last resort: cut every chunk_size characters
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split(text, chunk_size, rest)

    parts = text.split(sep)
    pieces = []
    for i, part in enumerate(parts):
        # keep the separator attached so no text is lost
        piece = part + (sep if i < len(parts) - 1 else "")
        pieces.extend(_split(piece, chunk_size, rest))
    return pieces


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks using recursive splitting."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    pieces = _split(text, chunk_size, SEPARATORS)

    chunks = []
    current, current_len = [], 0
    for piece in pieces:
        if current_len + len(piece) > chunk_size and current:
            chunks.append("".join(current).strip())

            # keep the last few pieces as overlap for the next chunk
            kept, kept_len = [], 0
            for p in reversed(current):
                if kept_len + len(p) > overlap:
                    break
                kept.insert(0, p)
                kept_len += len(p)
            while kept and kept_len + len(piece) > chunk_size:
                kept_len -= len(kept.pop(0))
            current, current_len = kept, kept_len

        current.append(piece)
        current_len += len(piece)

    last = "".join(current).strip()
    if last:
        chunks.append(last)
    return chunks