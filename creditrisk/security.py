"""API keys with scopes, expiry, revocation and per-key rate limits.

A key looks like `crk_<id>_<secret>`. Only the SHA-256 of the secret is stored; the id lets the server find the
record without comparing against every key. Secrets are 32 random bytes, so a fast hash is enough (there is
nothing to brute-force), and the comparison is constant-time.

Manage keys from the command line:

    python -m creditrisk.security create --name loan-portal --scopes score,letter --expires 2027-06-30
    python -m creditrisk.security list
    python -m creditrisk.security revoke <id>
"""

import argparse
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from creditrisk.config import SETTINGS

PREFIX = "crk"
SCOPES = {
    "score": "POST /score and /score/batch",
    "letter": "decision letters on /score (calls the LLM)",
    "monitor": "POST /monitor/drift",
    "model": "GET /model",
}


class AuthError(Exception):
    def __init__(self, status_code: int, detail: str, headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


def digest(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


@dataclass
class ApiKey:
    id: str
    name: str
    scopes: list[str]
    sha256: str
    created: str = field(default_factory=lambda: date.today().isoformat())
    expires: str | None = None
    disabled: bool = False
    rate_per_minute: int = 60

    def expired(self, today: date | None = None) -> bool:
        return self.expires is not None and (today or date.today()) > date.fromisoformat(self.expires)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "scopes": sorted(self.scopes), "expires": self.expires,
                "rate_per_minute": self.rate_per_minute}


def new_key(name: str, scopes: list[str], expires: str | None = None, rate_per_minute: int = 60) -> tuple[str, ApiKey]:
    unknown = set(scopes) - set(SCOPES)
    if unknown:
        raise ValueError(f"Unknown scope(s): {', '.join(sorted(unknown))}. Choose from {', '.join(SCOPES)}")
    if expires:
        date.fromisoformat(expires)
    key_id, secret = secrets.token_hex(4), secrets.token_urlsafe(32)
    record = ApiKey(key_id, name, sorted(set(scopes)), digest(secret), expires=expires, rate_per_minute=rate_per_minute)
    return f"{PREFIX}_{key_id}_{secret}", record


class RateLimiter:
    """Sliding one-minute window per key."""

    def __init__(self, window: float = 60.0, clock=time.monotonic):
        self.window, self.clock = window, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key_id: str, limit: int) -> None:
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key_id, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= limit:
                retry = int(self.window - (now - hits[0])) + 1
                raise AuthError(429, f"Rate limit of {limit} requests per minute exceeded; retry in {retry}s",
                                {"Retry-After": str(retry)})
            hits.append(now)


class KeyStore:
    """Keys from a JSON file, re-read when the file changes so revocation needs no restart."""

    def __init__(self, keys: list[ApiKey] | None = None, path: Path | None = None, limiter: RateLimiter | None = None):
        self.path = path
        self.limiter = limiter or RateLimiter()
        self._keys = {k.id: k for k in keys or []}
        self._mtime: float | None = None

    @classmethod
    def from_settings(cls) -> "KeyStore":
        return cls(path=SETTINGS.paths.api_keys)

    def _refresh(self) -> None:
        if self.path is None:
            return
        mtime = self.path.stat().st_mtime if self.path.exists() else None
        if mtime != self._mtime:
            self._keys = {k.id: k for k in load(self.path)}
            self._mtime = mtime

    def configured(self) -> bool:
        self._refresh()
        return bool(self._keys)

    def authenticate(self, presented: str | None) -> ApiKey:
        self._refresh()
        if not self._keys:
            raise AuthError(503, "No API keys are configured; create one with `python -m creditrisk.security create`")
        if not presented:
            raise AuthError(401, "Missing API key; send it in the X-API-Key header")
        parts = presented.split("_", 2)
        record = self._keys.get(parts[1]) if len(parts) == 3 and parts[0] == PREFIX else None
        matches = hmac.compare_digest(record.sha256 if record else digest(""), digest(parts[-1]))
        if record is None or not matches:
            raise AuthError(401, "Invalid API key")
        if record.disabled or record.expired():
            raise AuthError(401, "API key revoked or expired")
        self.limiter.check(record.id, record.rate_per_minute)
        return record


def load(path: Path) -> list[ApiKey]:
    if not path.exists():
        return []
    return [ApiKey(**k) for k in json.loads(path.read_text(encoding="utf-8"))["keys"]]


def save(path: Path, keys: list[ApiKey]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"keys": [asdict(k) for k in keys]}, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m creditrisk.security", description="Manage scoring API keys")
    parser.add_argument("--file", type=Path, default=SETTINGS.paths.api_keys)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="create a key and print it once")
    create.add_argument("--name", required=True)
    create.add_argument("--scopes", required=True, help=f"comma-separated: {','.join(SCOPES)}")
    create.add_argument("--expires", help="last valid day, YYYY-MM-DD")
    create.add_argument("--rate", type=int, default=60, help="requests per minute")
    sub.add_parser("list")
    revoke = sub.add_parser("revoke")
    revoke.add_argument("id")
    args = parser.parse_args(argv)

    keys = load(args.file)
    if args.command == "create":
        key, record = new_key(args.name, args.scopes.split(","), args.expires, args.rate)
        save(args.file, [*keys, record])
        print(f"Created key {record.id} for {record.name} with scopes {','.join(record.scopes)}.")
        print("Store it now; it is not saved and cannot be shown again:\n")
        print(key)
    elif args.command == "list":
        for k in keys:
            state = "revoked" if k.disabled else "expired" if k.expired() else "active"
            print(f"{k.id}  {k.name:20s} {','.join(k.scopes):28s} expires={k.expires or '-':10s} {state}")
    else:
        match = [k for k in keys if k.id == args.id]
        if not match:
            raise SystemExit(f"No key with id {args.id}")
        match[0].disabled = True
        save(args.file, keys)
        print(f"Revoked {args.id} ({match[0].name})")


if __name__ == "__main__":
    main()
