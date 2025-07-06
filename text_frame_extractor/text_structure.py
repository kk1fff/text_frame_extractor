class TextStructureAnalyzer:
    """Very small text structure analyzer that splits into lines."""

    def analyze(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
