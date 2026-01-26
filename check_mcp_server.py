import asyncio
import httpx
import sys

async def check_server(url):
    print(f"🔍 Checking URL: {url}")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 1. GET 요청 시도 (Health check)
            try:
                response = await client.get(url)
                print(f"  ✅ GET Request: Status {response.status_code}")
                print(f"     Headers: {dict(response.headers)}")
                print(f"     Content preview: {response.text[:100]}")
            except httpx.RequestError as e:
                print(f"  ❌ GET Request Failed: {e}")

            # 2. OPTIONS 요청 시도 (CORS 등 확인)
            try:
                response = await client.options(url)
                print(f"  ✅ OPTIONS Request: Status {response.status_code}")
            except httpx.RequestError as e:
                print(f"  ❌ OPTIONS Request Failed: {e}")

    except Exception as e:
        print(f"❌ Connection Error: {e}")
    print("-" * 50)

async def main():
    print("🚀 MCP Server Diagnostics Tool\n")
    
    # 1. 현재 설정된 URL 확인 (원격)
    target_url = "http://165.213.69.30:8001/mcp"
    await check_server(target_url)

    # 2. Localhost 확인 (우선순위 높음)
    print("\n[Localhost Check]")
    localhost_url = "http://localhost:8001/mcp"
    await check_server(localhost_url)
    
    # POST 요청 테스트 추가 (streamable-http는 POST 사용)
    print("\n[POST Request Test]")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # 빈 JSON-RPC 요청 시도
            dummy_payload = {
                "jsonrpc": "2.0",
                "method": "list_tools",  # MCP 표준 메서드
                "id": 1
            }
            print(f"🔍 Sending POST to {localhost_url}...")
            response = await client.post(localhost_url, json=dummy_payload)
            print(f"  ✅ POST Response: {response.status_code}")
            print(f"     Content: {response.text[:100]}")
    except Exception as e:
        print(f"  ❌ POST Failed: {e}")
    
    localhost_root = "http://localhost:8001/"
    await check_server(localhost_root)

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
