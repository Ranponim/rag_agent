import requests
import json

def test_mcp_with_requests():
    url = "http://165.213.69.30:8001/mcp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "Connection": "keep-alive"
    }
    
    payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "id": 1,
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "requests-checker", "version": "1.0.0"}
        }
    }
    
    print(f"🚀 [Requests] Connecting to {url}...")
    try:
        # PowerShell과 동일하게 POST 요청
        response = requests.post(
            url, 
            headers=headers, 
            json=payload, 
            timeout=30,
            stream=True # SSE 대응
        )
        
        print(f"✅ Status Code: {response.status_code}")
        print(f"📝 Response Headers: {dict(response.headers)}")
        
        # 첫 번째 라인(SSE 데이터) 확인
        for line in response.iter_lines():
            if line:
                print(f"📊 Received: {line.decode('utf-8')}")
                break # 하나만 받으면 성공
                
    except Exception as e:
        print(f"❌ Requests failed: {e}")

if __name__ == "__main__":
    test_mcp_with_requests()
