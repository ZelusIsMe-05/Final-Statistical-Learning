import re


def is_answer_grounded(answer: str, context: str, threshold: float = 0.25) -> bool:
    """
    Check whether an answer is grounded in the retrieved context.

    The check uses a lightweight keyword-overlap heuristic. It is not a
    substitute for semantic entailment, but it is useful as a low-cost guard for
    obvious hallucinations.
    """

    stopwords = {
        "là", "và", "của", "có", "trong", "với", "được", "các", "một", "những",
        "này", "đó", "để", "từ", "cho", "về", "như", "không", "tôi", "bạn",
        "gì", "nào", "bao", "nhiêu", "khi", "nếu", "thì", "mà", "vì", "do",
        "đã", "sẽ", "đang", "rất", "cũng", "hay", "hoặc", "thế", "nên",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "of", "in",
    }

    answer_words = set(re.findall(r"\w+", answer.lower())) - stopwords
    context_words = set(re.findall(r"\w+", context.lower()))

    if not answer_words:
        return False

    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap >= threshold
