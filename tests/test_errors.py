import json

from src.errors import invalid_key_response


def test_openai_shaped_403():
    resp = invalid_key_response({"reason": "no_credits", "message": "Usage window limit reached and no extra credits available."})
    assert resp.status_code == 403
    body = json.loads(resp.body)
    assert body == {
        "error": {
            "message": "Usage window limit reached and no extra credits available.",
            "type": "invalid_request_error",
            "code": "no_credits",
        }
    }


def test_missing_fields_fall_back():
    body = json.loads(invalid_key_response({}).body)
    assert body["error"]["message"]
    assert body["error"]["code"] == "forbidden"
