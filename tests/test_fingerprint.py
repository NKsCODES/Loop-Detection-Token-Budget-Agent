from loopguard.guards.fingerprint import canonical_args, make_record, normalize_output


def test_canonical_args_ignores_key_order():
    assert canonical_args({"b": 1, "a": "x"}) == canonical_args({"a": "x", "b": 1})


def test_normalize_masks_uuid_and_timestamp():
    a = "ERROR 503 (request_id=4a9ef8b7-89a4-5069-a374-0c4caf893fdd, at 2026-11-18T10:00:14Z)"
    b = "ERROR 503 (request_id=39ef8f10-aede-5414-8f92-a53b2c17454d, at 2026-11-18T10:00:21Z)"
    assert normalize_output(a) == normalize_output(b)


def test_echoed_free_text_query_is_masked():
    a = normalize_output('No results found for "pet insurance".', {"query": "pet insurance"})
    b = normalize_output('No results found for "dog cover plan".', {"query": "dog cover plan"})
    assert a == b


def test_identifiers_are_not_masked():
    a = normalize_output("job-1004: status=DONE", {"job_id": "job-1004"})
    b = normalize_output("job-1005: status=DONE", {"job_id": "job-1005"})
    assert a != b


def test_record_fingerprints():
    r1 = make_record(1, "fetch_weather", {"city": "Chennai"}, "ERROR 503 at 2026-11-18T10:00:00Z")
    r2 = make_record(2, "fetch_weather", {"city": "Chennai"}, "ERROR 503 at 2026-11-18T10:00:07Z")
    r3 = make_record(3, "fetch_weather", {"city": "Pune"}, "ERROR 503 at 2026-11-18T10:00:07Z")
    assert r1.call_fp == r2.call_fp
    assert r1.args_fp != r3.args_fp
