DECOMPOSE_SYSTEM = """You split a biomedical research question into focused, independently searchable sub-questions.
Rules: use ONLY entities and concepts present in the question or in the already-found evidence titles; do not invent
entities or assumptions; preserve the original intent; do not repeat sub-questions already covered; produce at most N
sub-questions. Return ONLY JSON: {"sub_questions": ["...", "..."]}"""


def decompose_user(question: str, max_n: int, covered: list[str], titles: list[str]) -> str:
    return (f"Question: {question}\nN = {max_n}\nAlready covered sub-questions: {covered or 'none'}\n"
            f"Titles of evidence found so far: {titles or 'none'}")
