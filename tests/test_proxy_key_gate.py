from src.proxy import bearer_token


def test_bearer_token_strips_scheme_case_insensitively():
    assert bearer_token("Bearer abc123") == "abc123"
    assert bearer_token("bearer abc123") == "abc123"


def test_bearer_token_passthrough_without_scheme():
    assert bearer_token("abc123") == "abc123"
