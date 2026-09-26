import urllib.request
import urllib.error
import json

base_url = "http://127.0.0.1:8000"

for path in ['/api/health', '/api/samples?limit=9', '/api/samples']:
    url = base_url + path
    try:
        req = urllib.request.urlopen(url)
        data = req.read()
        print(f"SUCCESS: {path} -> HTTP {req.status}")
        try:
            parsed = json.loads(data)
            if "samples" in parsed:
                print(f"  Returned {len(parsed['samples'])} samples.")
                print(f"  Classes: {[s['class_name'] for s in parsed['samples']]}")
            else:
                print(f"  Body: {str(parsed)[:100]}")
        except Exception:
            print(f"  Body (raw): {data[:100]}")
    except urllib.error.HTTPError as e:
        body = e.read()
        print(f"ERROR: {path} -> HTTP {e.code}: {body.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"FAILED: {path} -> {e}")
