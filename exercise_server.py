
# test the mcp server

import urllib.request, json, base64, os
h = {'Content-Type':'application/json', 'Accept':'application/json, text/event-stream'}

def call(method, params=None, sid=None, cid=1):
    body = {'jsonrpc':'2.0','method':method,'id':cid}
    if params: body['params'] = params
    data = json.dumps(body).encode()
    hdrs = dict(h)
    if sid: hdrs['Mcp-Session-Id'] = sid
    req = urllib.request.Request('http://host.containers.internal:8600/', data=data, headers=hdrs)
    r = urllib.request.urlopen(req, timeout=600)
    raw = r.read().decode()
    sess = r.headers.get('MCP-Session-Id', sid)
    for line in raw.split('\n'):
        if line.startswith('data: '):
            return json.loads(line[6:]), sess
    return json.loads(raw), sess

_, sid = call('initialize', {'protocolVersion':'2024-11-05','capabilities':{},'clientInfo':{'name':'test','version':'1.0'}})

# Remove old jar
print("Removing old jar...")
resp, _ = call('tools/call', {'name':'remove_jar','arguments':{'name':'magicdraw'}}, sid=sid, cid=2)
text = resp.get('result',{}).get('content',[{}])[0].get('text','') if 'content' in resp.get('result',{}) else json.dumps(resp)
print(text)

# Re-add
jar_path = "/workspace/md-javadoc-2026.0.0-110-f3252999-javadoc.jar"
print(f"\nEncoding {os.path.getsize(jar_path)/1024/1024:.1f} MB JAR...")
with open(jar_path, "rb") as f:
    content_b64 = base64.b64encode(f.read()).decode()
print("Adding JAR...")
resp, _ = call('tools/call', {'name':'add_jar','arguments':{'name':'magicdraw','content':content_b64}}, sid=sid, cid=3)
text = resp.get('result',{}).get('content',[{}])[0].get('text','') if 'content' in resp.get('result',{}) else json.dumps(resp)
print(text)

# List jars
print("\nListing JARs...")
resp, _ = call('tools/call', {'name':'list_jars','arguments':{}}, sid=sid, cid=4)
text = resp.get('result',{}).get('content',[{}])[0].get('text','') if 'content' in resp.get('result',{}) else json.dumps(resp)
print(text)
