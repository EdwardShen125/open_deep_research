"""Network isolation test - Phase 0.5 acceptance.

Run inside odr-pipeline-shell container. This container is on:
  - inside (can reach postgres/redis/searxng)
  - no-internet (internal: true, cannot reach external networks)

Expected results:
  - EXTERNAL bing.com:443        -> BLOCKED
  - EXTERNAL google.com:443      -> BLOCKED
  - INTERNAL odr-searxng:8080    -> OK
  - INTERNAL odr-postgres:5432   -> OK
  - INTERNAL odr-redis:6379      -> OK
"""
import socket
import urllib.request

def test_external(host, port=443):
    try:
        s = socket.create_connection((host, port), timeout=4)
        s.close()
        return f"REACHABLE (LEAK!)"
    except Exception as e:
        return f"BLOCKED ({type(e).__name__}: {str(e)[:40]})"

def test_internal_http(host, port=8080):
    try:
        req = urllib.request.Request(f"http://{host}:{port}/")
        r = urllib.request.urlopen(req, timeout=4)
        return f"OK HTTP {r.status}"
    except Exception as e:
        return f"FAIL ({type(e).__name__}: {str(e)[:60]})"

def test_internal_tcp(host, port):
    try:
        s = socket.create_connection((host, port), timeout=4)
        s.close()
        return f"OK TCP open"
    except Exception as e:
        return f"FAIL ({type(e).__name__}: {str(e)[:40]})"

cases = [
    ("EXTERNAL bing.com:443",        test_external("bing.com")),
    ("EXTERNAL google.com:443",      test_external("google.com")),
    ("EXTERNAL github.com:443",      test_external("github.com")),
    ("INTERNAL odr-searxng:8080",    test_internal_http("odr-searxng", 8080)),
    ("INTERNAL odr-postgres:5432",   test_internal_tcp("odr-postgres", 5432)),
    ("INTERNAL odr-redis:6379",      test_internal_tcp("odr-redis", 6379)),
]

print("=" * 70)
print(" PHASE 0.5 NETWORK ISOLATION TEST")
print("=" * 70)
for label, result in cases:
    is_external = label.startswith("EXTERNAL")
    if is_external:
        ok = "BLOCKED" in result
        flag = "PASS" if ok else "FAIL"
    else:
        ok = "OK" in result
        flag = "PASS" if ok else "FAIL"
    print(f" [{flag}] {label:42s} {result}")
print("=" * 70)

external_pass = all(("BLOCKED" in c[1]) for c in cases if c[0].startswith("EXTERNAL"))
internal_pass = all(("OK" in c[1]) for c in cases if c[0].startswith("INTERNAL"))
print(f"\nSummary: external_blocked={external_pass}, internal_reachable={internal_pass}")
print("RESULT:", "ALL_PASS" if (external_pass and internal_pass) else "ISOLATION_BROKEN")
