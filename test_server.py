#!/usr/bin/env python3
"""Test all MCP tools against whatever jars are already indexed."""

import urllib.request, json, time, os, sys
sys.stdout.reconfigure(line_buffering=True)

BASE = 'http://host.containers.internal:8600'
h = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
passed = 0
failed = 0

def call(method, params=None, sid=None, cid=1):
    body = {'jsonrpc': '2.0', 'method': method, 'id': cid}
    if params: body['params'] = params
    req = urllib.request.Request(f'{BASE}/', data=json.dumps(body).encode(), headers=h)
    if sid: req.add_header('Mcp-Session-Id', sid)
    r = urllib.request.urlopen(req, timeout=600)
    raw = r.read().decode()
    sess = r.headers.get('MCP-Session-Id', sid)
    for line in raw.split('\n'):
        if line.startswith('data: '):
            return json.loads(line[6:]), sess
    return json.loads(raw), sess

def tool(name, args, sid):
    resp, _ = call('tools/call', {'name': name, 'arguments': args}, sid=sid)
    content = resp.get('result', {}).get('content', [{}])
    return content[0].get('text', '') if content else json.dumps(resp)

def check(label, got, detail=''):
    global passed, failed
    ok = bool(got) if not isinstance(got, bool) else got
    status = "PASS" if ok else "FAIL"
    if ok: passed += 1
    else: failed += 1
    extra = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{extra}", flush=True)

# ── Init ───────────────────────────────────────────────────────────
print("\n=== Initialize ===", flush=True)
resp, sid = call('initialize', {
    'protocolVersion': '2024-11-05', 'capabilities': {},
    'clientInfo': {'name': 'test', 'version': '1.0'}
})
check("initialize returns session", bool(sid))

# ── list_jars ─────────────────────────────────────────────────────
print("\n=== list_jars ===", flush=True)
t = tool('list_jars', {}, sid)
print(t, flush=True)
has_jars = 'No JARs' not in t and 'Indexed JARs' in t
check("at least one jar indexed", has_jars)
if not has_jars:
    print("  STOP — no jars to test against", flush=True)
    print(f"\n{'='*50}\n  {passed} passed, {failed} failed\n{'='*50}", flush=True)
    sys.exit(1 if failed else 0)

# Extract jar name: first non-empty line that doesn't start with space/Indexed
jar_name = ''
for l in t.split('\n'):
    stripped = l.strip()
    if stripped and not stripped.startswith('Indexed') and not stripped.startswith('status') and not stripped.startswith('hash') and not stripped.startswith('added') and not stripped.startswith('symbol') and not stripped.startswith('error'):
        jar_name = stripped
        break
print(f"  Using jar: {jar_name}", flush=True)

# ── jar_status ────────────────────────────────────────────────────
print("\n=== jar_status ===", flush=True)
t = tool('jar_status', {'name': jar_name}, sid)
print(t, flush=True)
check("status shows name", jar_name in t)
check("status is indexed or indexing", 'indexed' in t.lower() or 'indexing' in t.lower())

if 'indexing' in t.lower():
    print("  Jar still indexing — query tests may be incomplete", flush=True)

# ── lookup_symbol (exact match) ───────────────────────────────────
print("\n=== lookup_symbol (exact) ===", flush=True)
t = tool('lookup_symbol', {'fqn': 'com.nomagic.uml2.ext.jmi.helpers.StereotypesHelper'}, sid)
check("finds StereotypesHelper", 'StereotypesHelper' in t, t[:80])
check("shows kind", 'class' in t.lower())
check("shows package", 'com.nomagic.uml2.ext.jmi.helpers' in t)

# ── lookup_symbol (not found, suggests) ───────────────────────────
print("\n=== lookup_symbol (not found, with suggestion) ===", flush=True)
t = tool('lookup_symbol', {'fqn': 'com.nomagic.uml2.ext.jmi.helpers.StereotypeHelper'}, sid)
print(f"  Response: {t[:200]}", flush=True)
check("reports not found", 'not found' in t.lower())
check("suggestion when available", 'Did you mean' in t or 'not found' in t)

# ── lookup_symbol (method) ────────────────────────────────────────
print("\n=== lookup_symbol (method) ===", flush=True)
t = tool('lookup_symbol', {'fqn': 'com.nomagic.uml2.ext.jmi.helpers.StereotypesHelper#getStereotypes'}, sid)
check("finds method", 'getStereotypes' in t)

# ── search_docs keyword ───────────────────────────────────────────
print("\n=== search_docs (keyword) ===", flush=True)
t = tool('search_docs', {'query': 'getAllStereotypes', 'limit': 5}, sid)
print(t[:300], flush=True)
check("returns results", 'Top' in t)
check("contains expected", 'getAllStereotypes' in t)

# ── search_docs with jar_filter ───────────────────────────────────
print("\n=== search_docs (jar filter) ===", flush=True)
t = tool('search_docs', {'query': 'Stereotype', 'jar_filter': jar_name, 'limit': 3}, sid)
check("filters by jar", 'Top' in t)

# ── list_packages ─────────────────────────────────────────────────
print("\n=== list_packages ===", flush=True)
t = tool('list_packages', {}, sid)
pkgs = [l for l in t.split('\n') if l.strip()]
check("returns packages", len(pkgs) > 5)
print(f"  {len(pkgs)} packages, first: {pkgs[0]}", flush=True)

# ── list_packages with jar_filter ─────────────────────────────────
print("\n=== list_packages (jar filter) ===", flush=True)
t = tool('list_packages', {'jar_filter': jar_name}, sid)
filtered = [l for l in t.split('\n') if l.strip()]
check("filtered returns packages", len(filtered) > 0)

first_pkg = filtered[0] if filtered else 'com.nomagic'

# ── list_classes ──────────────────────────────────────────────────
print("\n=== list_classes ===", flush=True)
t = tool('list_classes', {'package': first_pkg}, sid)
check(f"lists classes in {first_pkg}", len([l for l in t.split('\n') if l.strip()]) > 0)

# ── list_classes with jar_filter ──────────────────────────────────
print("\n=== list_classes (jar filter) ===", flush=True)
t = tool('list_classes', {'package': first_pkg, 'jar_filter': jar_name}, sid)
check("filtered classes", len([l for l in t.split('\n') if l.strip()]) > 0)

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'='*50}", flush=True)
print(f"  {passed} passed, {failed} failed out of {passed+failed} tests", flush=True)
print(f"{'='*50}", flush=True)
sys.exit(0 if failed == 0 else 1)
