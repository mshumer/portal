import os
import asyncio
import aiohttp
import requests
import json
import sys
import traceback
from flask import Flask, render_template, request, redirect, jsonify, Response, stream_with_context
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

CORS(app)

SERPAPI_API_KEY = "YOUR_SERPAPI_KEY"
openai_api_key = "YOUR_OPENAI_KEY"


def rerank_results(query, candidates):
    """Rerank search results using OpenAI to select the best match."""
    # Build the options string with numbered IDs
    options = ""
    for candidate in candidates:
        options += f"Option {candidate['index']}:\n{candidate['title']}\n{candidate['snippet']}\n{candidate['link']}\n\n"

    # Construct the prompt
    prompt = f"""You are an assistant that selects the most relevant web page content based on a user's query.

User Query:
{query}

Below are several web page contents labeled with option numbers:

{options}
Based on the user query, choose the most relevant option number from the list above. Provide only the **number** of the chosen option."""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        data = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        }

        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers,
                                 json=data).json()

        # Extract the AI's response
        choice_text = response['choices'][0]['message']['content'].strip(
        ).replace('Option ', '').replace('#', '')
        chosen_option = int(choice_text)
        # Find the candidate with the matching index
        best_candidate = next(
            (c for c in candidates if c['index'] == chosen_option), None)
        if best_candidate:
            print(
                f"🏁 AI selected option {chosen_option}: {best_candidate['title']}"
            )
            return best_candidate  # Return the best candidate directly
        else:
            print("No valid option selected by the AI.")
            return None
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return None


async def determine_query_intent(query):
    """Determine if the query is seeking navigation or information."""
    prompt = f"""Analyze this search query and determine if the user is:
1. Looking to navigate to a specific website/service (NAVIGATION)
2. Asking a question that needs an informative answer (ANSWER)

Query: "{query}"

Respond with exactly one word: either NAVIGATION or ANSWER"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    data = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": prompt
        }]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions",
                                headers=headers,
                                json=data) as response:
            result = await response.json()
            intent = result['choices'][0]['message']['content'].strip()
            return intent == "NAVIGATION"


async def get_search_results_async(query, limit=10):
    """Asynchronous version of get_search_results"""
    serpapi_url = 'https://serpapi.com/search'
    params = {
        'q': query,
        'api_key': SERPAPI_API_KEY,
        'engine': 'google',
        'num': limit
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(serpapi_url, params=params) as response:
            if response.status != 200:
                return []
            results = await response.json()

    organic_results = results.get('organic_results', [])
    candidates = []
    for idx, item in enumerate(organic_results):
        candidates.append({
            "index":
            idx + 1,
            "title":
            item.get("title"),
            "snippet":
            item.get("snippet"),
            "link":
            item.get("link"),
            "content":
            f"{item.get('title', '')}\n{item.get('snippet', '')}"
        })
    return candidates


def generate_chunks(query, candidates):
    """Generate streaming response chunks"""
    try:
        # Send initial message
        # yield "data: {\"content\": \"Analyzing search results...\"}\n\n"

        context = "\n\n".join([
            f"Source {c['index']}:\nTitle: {c['title']}\nContent: {c['snippet']}"
            for c in candidates[:5]
        ])

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }

        data = {
            "model":
            "gpt-4o-mini",
            "messages": [{
                "role":
                "user",
                "content":
                f"Based on these search results, provide a clear answer about: {query}\n\nResults:\n{context}"
            }],
            "stream":
            True
        }

        # Make request to OpenAI
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers,
                                 json=data,
                                 stream=True)

        print(f"OpenAI Status: {response.status_code}")

        if response.status_code != 200:
            yield f'data: {{"content": "Error: Unable to generate response"}}\n\n'
            return

        # Process the streaming response
        for line in response.iter_lines():
            if line:
                try:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        if json_str == '[DONE]':
                            # Send completion event
                            yield 'event: done\ndata: {"status": "completed"}\n\n'
                            break

                        json_data = json.loads(json_str)
                        if content := json_data.get('choices', [{}])[0].get(
                                'delta', {}).get('content'):
                            chunk = f'data: {{"content": {json.dumps(content)}}}\n\n'
                            print(f"Sending chunk: {chunk[:100]}...")
                            yield chunk

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue

    except Exception as e:
        print(f"Error in generate_chunks: {e}")
        yield f'data: {{"content": "Error: {str(e)}"}}\n\n'
    finally:
        # Ensure completion event is sent even if there's an error
        yield 'event: done\ndata: {"status": "completed"}\n\n'


@app.route('/stream_answer')
def stream_answer():
    query = request.args.get('query')
    candidates_json = request.args.get('candidates')

    print(f"Stream request received - Query: {query[:50]}...")

    if not query or not candidates_json:
        return "Missing parameters", 400

    try:
        candidates = json.loads(candidates_json)
    except json.JSONDecodeError:
        return "Invalid candidates data", 400

    # Create the streaming response
    return Response(stream_with_context(generate_chunks(query, candidates)),
                    mimetype='text/event-stream',
                    headers={
                        'Cache-Control': 'no-cache',
                        'X-Accel-Buffering': 'no',
                        'Connection': 'keep-alive',
                        'Content-Type': 'text/event-stream; charset=utf-8',
                        'Access-Control-Allow-Origin': '*'
                    })


@app.route('/', methods=['GET', 'POST'])
async def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return render_template('index.html',
                                   error='Please enter a search query.')

        try:
            # Run intent classification and search results retrieval in parallel
            intent_task = asyncio.create_task(determine_query_intent(query))
            search_results_task = asyncio.create_task(
                get_search_results_async(query, limit=10))

            # Wait for both tasks to complete
            is_navigation, candidates = await asyncio.gather(
                intent_task, search_results_task)

            if not candidates:
                return render_template('index.html',
                                       error='No search results found.')

            if is_navigation:
                # Use existing rerank_results function for navigation
                best_result = rerank_results(query, candidates)
                if not best_result:
                    return render_template(
                        'index.html',
                        error='AI failed to select the best result.')
                return redirect(best_result['link'])
            else:
                # Return the template with candidates for streaming
                return render_template('index.html',
                                       query=query,
                                       stream_answer=True,
                                       candidates_json=json.dumps(candidates),
                                       sources=candidates[:3])

        except Exception as e:
            return render_template('index.html',
                                   error=f'An error occurred: {str(e)}')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
