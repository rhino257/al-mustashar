import asyncio
import httpx
import time

import uuid # Import uuid

# --- Configuration ---
# IMPORTANT: Replace these with your actual endpoint and a valid JWT
CHAT_ID_FOR_TEST = str(uuid.uuid4()) # Generate a random UUID for chat_id
API_ENDPOINT = f"http://localhost:8000/rag/query?query=hello&chat_id={CHAT_ID_FOR_TEST}"
USER_JWT = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlJFTjBtaTYxSXNmTjVueHkiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3FobWJvbWhzd2t4dG51c2FmcmpzLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiI0NGU3ZDIwMy1jMjhhLTQ2MTAtYjJkZi02ZTU5NDhmY2QzM2IiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzQ5NjUyODkzLCJpYXQiOjE3NDk2NDkyOTMsImVtYWlsIjoibW9obWVkYWxhYnlkaDE3QGdtYWlsLmNvbSIsInBob25lIjoiIiwiYXBwX21ldGFkYXRhIjp7InByb3ZpZGVyIjoiZW1haWwiLCJwcm92aWRlcnMiOlsiZW1haWwiXX0sInVzZXJfbWV0YWRhdGEiOnsiZW1haWwiOiJtb2htZWRhbGFieWRoMTdAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsInBob25lX3ZlcmlmaWVkIjpmYWxzZSwic3ViIjoiNDRlN2QyMDMtYzI4YS00NjEwLWIyZGYtNmU1OTQ4ZmNkMzNiIn0sInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiYWFsIjoiYWFsMSIsImFtciI6W3sibWV0aG9kIjoicGFzc3dvcmQiLCJ0aW1lc3RhbXAiOjE3NDk2NDkyOTN9XSwic2Vzc2lvbl9pZCI6IjM0MjgyY2RlLTM0NDctNDRmNC04MjgzLThhYjhiOTdjODIzZSIsImlzX2Fub255bW91cyI6ZmFsc2V9.gcPHqLSN0RMZ8H8XLpYlc5_kXx6NK6akol1Kmtw8qyY"

REQUEST_COUNT = 12  # Number of requests to send
REQUEST_DELAY_SECONDS = 0.2  # Delay between requests (to send them quickly but not instantly)
# ---------------------

async def send_request(client: httpx.AsyncClient, request_num: int, endpoint_url: str, token: str):
    headers = {
        "Authorization": f"Bearer {token}"
    }
    try:
        print(f"Sending request #{request_num} to {endpoint_url}...")
        response = await client.get(endpoint_url, headers=headers, timeout=10.0)
        print(f"Request #{request_num}: Status Code = {response.status_code}")
        if response.status_code == 429:
            print(f"Request #{request_num}: Rate limit likely exceeded!")
        # You can print response.text if needed for debugging, but it might be long for SSE
        # print(f"Request #{request_num}: Response Text = {response.text[:200]}...") 
    except httpx.ReadTimeout:
        print(f"Request #{request_num}: Timed out.")
    except httpx.RequestError as e:
        print(f"Request #{request_num}: Request failed: {e}")
    except Exception as e:
        print(f"Request #{request_num}: An unexpected error occurred: {e}")

async def main():
    if API_ENDPOINT == "YOUR_API_ENDPOINT_HERE" or USER_JWT == "YOUR_VALID_JWT_HERE":
        print("ERROR: Please update API_ENDPOINT and USER_JWT in the script before running.")
        return

    print(f"Starting rate limit test: Sending {REQUEST_COUNT} requests to {API_ENDPOINT}")
    print(f"Simulating requests for user associated with the provided JWT.")
    print(f"Expected rate limit: 10 requests per minute.")
    print("---")

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(1, REQUEST_COUNT + 1):
            # Create the coroutine for each request and add it to the list
            task = send_request(client, i, API_ENDPOINT, USER_JWT)
            tasks.append(task)
        
        # Run all request tasks concurrently
        print(f"Launching {REQUEST_COUNT} requests in parallel...")
        await asyncio.gather(*tasks)

    print("---")
    print("Test finished.")
    print("Check the status codes. You should see some 200 OK responses followed by 429 Too Many Requests if the rate limit is working.")
    print("Remember that the 1-minute window for the rate limit starts from the first request that hits the counter.")

if __name__ == "__main__":
    asyncio.run(main())
