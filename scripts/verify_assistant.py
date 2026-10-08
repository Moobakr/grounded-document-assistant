"""Browser -> API -> SQLite -> mocked Groq transport -> rendered tutor answer.

Uses an isolated temporary workspace and synthetic responses, not a quality benchmark
or a live Groq request. No credentials or changes to the user's index are needed.
"""

import json
import io
import argparse
from pathlib import Path
import socket
import tempfile
import threading
import time
from unittest.mock import patch
from urllib.parse import urlsplit

import httpx
import numpy as np
from PIL import Image, ImageStat
from playwright.sync_api import expect, sync_playwright
import uvicorn

from scripts.verify_web import fixture_pdf, no_overflow
from src.groq_generation import GroqGenerator
from src.web import create_app


class FixtureEmbeddings:
    identity = "assistant-browser-fixture"

    def encode(self, texts):
        return np.array([[1, 0] for text in texts], dtype=np.float32)

    def split_text(self, text, size, overlap):
        return [text]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="chrome", choices=("chrome", "msedge", "chromium"))
    args = parser.parse_args()
    output = Path("data/assistant-verification")
    output.mkdir(parents=True, exist_ok=True)
    report = {"mocked_groq": True, "checks": [], "page_errors": [], "external_browser_requests": []}
    requests = []
    failure = {"enabled": False}

    def checked(label):
        report["checks"].append(label)
        print("Verified " + label, flush=True)

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if failure["enabled"]:
            return httpx.Response(429, json={"error": {"message": "private-test-key"}})
        prompt = body["messages"][0]["content"]
        if "Create 3 short practice" in prompt:
            answer = "## Practice questions\n1. What is uploaded? [1]\n2. What happens after uploading? [1]\n3. Is indexing mentioned? [1]\n## Answer key\n1. A PDF. [1]\n2. Local indexing. [1]\n3. Yes. [1]"
        elif "Summarize the requested" in prompt:
            answer = "## Topic summary\nThe fixture mentions local upload and indexing. [1]"
        else:
            answer = "## Local upload\n**Upload** places your PDF into the local indexing workflow. [1]\nIllustrative example (not from the PDF): organize a small lecture collection.\n<img src=x onerror=alert(1)>"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
            "answer": answer, "source_ids": [1], "abstained": False})}}]})

    with tempfile.TemporaryDirectory() as temporary:
        app = create_app(root=temporary, embedder=FixtureEmbeddings())
        app.state.workspace.groq_key = ""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started and worker.is_alive() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert server.started, "Temporary verification server failed to start."
            with patch("src.web.GroqGenerator", side_effect=lambda key, model: GroqGenerator(key, model, httpx.MockTransport(respond))):
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(channel=args.channel, headless=True)
                    try:
                        for width, height, name in ((1440, 1000, "desktop"), (390, 844, "mobile")):
                            context = browser.new_context(viewport={"width": width, "height": height}, accept_downloads=True)

                            def route_request(route):
                                if urlsplit(route.request.url).hostname != "127.0.0.1":
                                    report["external_browser_requests"].append(route.request.url)
                                    route.abort()
                                else:
                                    route.continue_()

                            context.route("**/*", route_request)
                            page = context.new_page()
                            page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
                            page.goto(url)
                            expect(page.locator("#operation-banner")).to_be_hidden(timeout=30000)
                            if name == "desktop":
                                page.locator("#file-input").set_input_files({"name": "assistant-fixture.pdf", "mimeType": "application/pdf", "buffer": fixture_pdf()})
                                expect(page.locator("#document-total")).to_have_text("1", timeout=30000)
                                expect(page.locator("#operation-banner")).to_be_hidden()
                            page.locator("#ai-settings").click()
                            expect(page.locator("#groq-key")).to_have_attribute("type", "password")
                            page.locator("#groq-key").fill("private-test-key")
                            page.locator("#connect-ai").click()
                            expect(page.locator("#ai-dialog")).to_be_hidden()
                            expect(page.locator("#groq-key")).to_have_value("")
                            expect(page.locator("#cloud-notice")).to_be_visible()
                            assert "private-test-key" not in context.request.get(url + "/api/ai").text()
                            assert page.evaluate("localStorage.length + sessionStorage.length") == 0
                            checked(name + ": memory-only key connection and cloud disclosure")

                            def query(question):
                                page.locator("#question").fill(question)
                                with page.expect_response(lambda response: response.url.endswith("/api/query")) as response:
                                    page.locator("#run-query").click()
                                data = response.value.json()
                                assert response.value.status == 200, data
                                expect(page.locator("#operation-banner")).to_be_hidden()
                                return data

                            result = query("Explain local upload and indexing")
                            assert result["answer"] and not result["generation_error"]
                            expect(page.locator("#answer-title")).to_have_text("Explanation")
                            expect(page.locator("#answer-content strong")).to_have_text("Upload")
                            expect(page.locator("#answer-content img")).to_have_count(0)
                            expect(page.locator("#answer-content")).to_contain_text("<img src=x")
                            page.locator(".citation-button").first.click()
                            expect(page.locator("#preview-page")).to_have_value("1")
                            expect(page.locator("#preview-image")).to_be_visible()
                            pixels = context.request.get(page.locator("#preview-image").get_attribute("src")).body()
                            with Image.open(io.BytesIO(pixels)) as image:
                                assert max(ImageStat.Stat(image).stddev) > 5, "PDF preview is blank."
                            page.locator("#toast").evaluate("element => element.hidden = true")
                            no_overflow(page)
                            page.screenshot(path=str(output / f"{name}-assistant.png"), full_page=True)
                            checked(name + ": grounded response, safe formatting, and nonblank cited PDF preview")

                            page.locator("#follow-up").click()
                            query("Give me another example")
                            history = json.loads(requests[-1]["messages"][1]["content"])["conversation"]
                            assert history and history[-1]["question"] == "Explain local upload and indexing"
                            checked(name + ": follow-up conversation reaches generation")

                            page.locator("#learning-task").select_option("quiz")
                            query("Test my understanding")
                            expect(page.locator("#answer-content")).to_contain_text("Answer key")
                            expect(page.locator("#answer-title")).to_have_text("Practice questions")
                            page.locator("#learning-task").select_option("summary")
                            query("Summarize the topic")
                            expect(page.locator("#answer-title")).to_have_text("Topic summary")
                            assert len(json.loads(requests[-1]["messages"][1]["content"])["conversation"]) == 2
                            with page.expect_download() as download:
                                page.locator("#download-results").click()
                            downloaded = output / f"{name}-result.json"
                            download.value.save_as(str(downloaded))
                            assert json.loads(downloaded.read_text())["provider"] == "groq"
                            no_overflow(page)
                            checked(name + ": quiz, summary, bounded history, and answer export")

                            failure["enabled"] = True
                            result = query("Explain again")
                            assert result["answer"] is None and result["sources"]
                            expect(page.locator("#query-error")).to_contain_text("limit")
                            expect(page.locator(".source-item")).to_have_count(1)
                            assert "private-test-key" not in page.locator("#query-error").inner_text()
                            failure["enabled"] = False
                            page.locator("#clear-query").click()
                            expect(page.locator("#conversation-count")).to_have_text("New conversation")
                            page.locator("#ai-settings").click()
                            page.locator("#disconnect-ai").click()
                            expect(page.locator("#ai-dialog")).to_be_hidden()
                            assert not context.request.get(url + "/api/ai").json()["configured"]
                            checked(name + ": rate-limit fallback, new conversation, and disconnect")
                            context.close()
                        assert not report["page_errors"], report["page_errors"]
                        assert not report["external_browser_requests"], report["external_browser_requests"]
                    finally:
                        browser.close()
        finally:
            server.should_exit = True
            worker.join(timeout=15)
            listener.close()
            assert not worker.is_alive(), "Verification server failed to stop."
    report["mocked_groq_requests"] = len(requests)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
