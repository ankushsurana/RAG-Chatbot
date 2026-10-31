from __future__ import annotations

from types import SimpleNamespace

from ragbot.api.app import RateLimiter, request_owners


def _request(**headers: str) -> SimpleNamespace:
    return SimpleNamespace(headers={k.replace("_", "-"): v for k, v in headers.items()})


class TestOwners:
    def test_public_visitor_acts_only_for_their_browser(self):
        request = _request(cf_connecting_ip="1.2.3.4", x_client_id="browser-123")
        assert request_owners(request) == ("browser-123",)

    def test_public_visitor_cannot_claim_older_local_chats(self):
        assert "" not in request_owners(_request(cf_connecting_ip="1.2.3.4"))

    def test_invalid_client_id_is_ignored(self):
        request = _request(cf_connecting_ip="1.2.3.4", x_client_id="x' OR 1=1")
        assert request_owners(request) == ("address-1.2.3.4",)

    def test_local_user_also_sees_older_chats(self):
        assert request_owners(_request(x_client_id="browser-123")) == ("browser-123", "")
        assert request_owners(_request()) == ("",)


class TestRateLimiter:
    def test_blocks_after_limit_and_recovers(self):
        limiter = RateLimiter(2)
        assert limiter.allow("a", now=0.0)
        assert limiter.allow("a", now=1.0)
        assert not limiter.allow("a", now=2.0)
        assert limiter.allow("a", now=61.0)

    def test_visitors_are_limited_separately(self):
        limiter = RateLimiter(1)
        assert limiter.allow("a", now=0.0)
        assert limiter.allow("b", now=0.0)
