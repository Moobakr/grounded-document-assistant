"""Browser-to-index smoke check; creates and removes only its own fixture PDF."""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageStat
from playwright.sync_api import sync_playwright, expect
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def fixture_pdf():
    writer = PdfWriter()
    page = writer.add_blank_page(width=500, height=300)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                            NameObject("/Subtype"): NameObject("/Type1"),
                                            NameObject("/BaseFont"): NameObject("/Helvetica")}),
    })})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 16 Tf 25 170 Td (Interface verification. Local upload and indexing.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def no_overflow(page):
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Horizontal overflow."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--channel", default="chrome")
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in ("127.0.0.1", "localhost"):
        raise ValueError("Browser verification only accepts the local interface.")
    output = Path("data/browser-verification")
    output.mkdir(parents=True, exist_ok=True)
    report = {"checks": [], "screenshots": [], "external_requests": [], "page_errors": []}

    def checked(label):
        report["checks"].append(label)
        print("Verified " + label, flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True,
                                      permissions=["clipboard-read", "clipboard-write"])

        def route_request(route):
            host = urlsplit(route.request.url).hostname
            if host not in ("127.0.0.1", "localhost"):
                report["external_requests"].append(route.request.url)
                route.abort()
            else:
                route.continue_()

        context.route("**/*", route_request)
        page = context.new_page()
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.goto(args.url)
        expect(page.locator("#operation-banner")).to_be_hidden(timeout=30000)
        expect(page.get_by_role("heading", name="Document library", exact=True)).to_be_visible()
        original_docs = context.request.get(args.url + "/api/documents").json()
        original_ids = {doc["id"] for doc in original_docs}
        assert original_docs, "The real document collection must be indexed first."
        expect(page.locator("#document-total")).to_have_text(str(len(original_docs)))
        expect(page.locator("#preview-image")).to_be_visible(timeout=30000)
        pixels = context.request.get(page.locator("#preview-image").get_attribute("src")).body()
        with Image.open(io.BytesIO(pixels)) as image:
            assert max(ImageStat.Stat(image).stddev) > 5, "PDF preview appears blank."
        no_overflow(page)
        page.screenshot(path=str(output / "desktop-library.png"), full_page=True)
        report["screenshots"].append("desktop-library.png")
        checked("existing collection and nonblank real PDF preview")

        page.locator("#document-filter").fill("Lecture 8")
        expect(page.locator("#document-rows tr")).to_have_count(1)
        page.locator("#document-filter").fill("")
        expect(page.locator("#document-rows tr")).to_have_count(len(original_docs))
        checked("document filtering")

        def query(question, mode="search"):
            page.locator("#mode-search" if mode == "search" else "#mode-answer").click()
            page.locator("#question").fill(question)
            with page.expect_response(lambda response: response.url.endswith("/api/query"), timeout=180000) as response:
                page.locator("#run-query").click()
            data = response.value.json()
            assert response.value.status == 200, data
            expect(page.locator("#operation-banner")).to_be_hidden(timeout=10000)
            return data

        fixture_id = None
        fixture_bytes = fixture_pdf()
        expected_fixture_id = hashlib.sha256(fixture_bytes).hexdigest()
        assert expected_fixture_id not in original_ids, "Verification fixture is already in the collection."
        try:
            page.locator("#nav-workspace").click()
            result = query("What does correlation mean?")
            assert any(match["chunk"]["page"] == 3 and "Lecture 1" in match["chunk"]["source"] for match in result["sources"])
            expect(page.locator(".source-item")).to_have_count(len(result["sources"]))
            page.locator(".source-item").first.click()
            expect(page.locator("#preview-page")).to_have_value(str(result["sources"][0]["chunk"]["page"]))
            no_overflow(page)
            page.screenshot(path=str(output / "desktop-search.png"), full_page=True)
            report["screenshots"].append("desktop-search.png")
            checked("real semantic search, page references, and source selection")

            series_id = next(doc["id"] for doc in original_docs if "Lecture 8" in doc["source"])
            page.locator("#document-scope").select_option(series_id)
            scoped = query("What is a time series?")
            assert scoped["sources"] and all(match["chunk"]["document_id"] == series_id for match in scoped["sources"])
            checked("document-scoped search")

            page.locator("#copy-results").click()
            expect(page.locator("#toast")).to_have_text("Passages copied.")
            with page.expect_download() as download:
                page.locator("#download-results").click()
            saved = output / "downloaded-results.json"
            download.value.save_as(str(saved))
            assert json.loads(saved.read_text(encoding="utf-8"))["sources"] == scoped["sources"]
            checked("clipboard and JSON download")

            page.locator("#mode-answer").click()
            page.locator("#ai-provider").select_option("local")
            model_online = context.request.get(args.url + "/api/model").json()["online"]
            answer = query("What is a time series?", "ask")
            assert answer["sources"]
            if model_online:
                assert answer["answer"] is not None and not answer["generation_error"]
                expect(page.locator("#answer-section")).to_be_visible()
                citation = page.locator(".citation-button").first
                if citation.count():
                    citation.click()
                checked("local answer rendering and citation inspection")
            else:
                assert answer["generation_error"] and answer["answer"] is None
                expect(page.locator("#query-error")).to_be_visible()
                expect(page.locator(".source-item").first).to_be_visible()
                checked("model-offline error retains retrieved passages")

            page.locator("#nav-library").click()
            pdf_bytes = fixture_bytes
            page.locator("#file-input").set_input_files({"name": "interface-verification.pdf", "mimeType": "application/pdf", "buffer": pdf_bytes})
            expect(page.locator("#document-total")).to_have_text(str(len(original_docs) + 1), timeout=60000)
            expect(page.locator("#operation-banner")).to_be_hidden(timeout=10000)
            documents = context.request.get(args.url + "/api/documents").json()
            fixture_id = next(doc["id"] for doc in documents if doc["source"] == "interface-verification.pdf")
            checked("PDF upload and persisted ingestion")

            page.locator("#file-input").set_input_files({"name": "interface-verification-renamed.pdf", "mimeType": "application/pdf", "buffer": pdf_bytes})
            expect(page.locator("#upload-results")).to_contain_text("Already indexed", timeout=60000)
            expect(page.locator("#operation-banner")).to_be_hidden(timeout=10000)
            expect(page.locator("#document-total")).to_have_text(str(len(original_docs) + 1))
            checked("duplicate upload does not add a document")

            page.locator("#search-document").click()
            uploaded = query("What does the interface verification document say?")
            assert uploaded["sources"] and all(match["chunk"]["document_id"] == fixture_id for match in uploaded["sources"])
            checked("uploaded PDF is immediately searchable")

            page.locator("#nav-library").click()
            page.get_by_role("button", name="Delete interface-verification.pdf", exact=True).click()
            expect(page.locator("#delete-dialog")).to_be_visible()
            page.get_by_role("button", name="Cancel", exact=True).click()
            expect(page.locator("#document-total")).to_have_text(str(len(original_docs) + 1))
            page.get_by_role("button", name="Delete interface-verification.pdf", exact=True).click()
            page.locator("#confirm-delete").click()
            expect(page.locator("#document-total")).to_have_text(str(len(original_docs)), timeout=30000)
            expect(page.locator("#operation-banner")).to_be_hidden(timeout=10000)
            fixture_id = None
            checked("cancel and confirm document deletion")

            page.locator("#file-input").set_input_files({"name": "not-a-pdf.txt", "mimeType": "text/plain", "buffer": b"invalid"})
            expect(page.locator("#upload-results")).to_contain_text("Only PDF files", timeout=10000)
            expect(page.locator("#operation-banner")).to_be_hidden(timeout=10000)
            checked("invalid upload error state")
        finally:
            documents = context.request.get(args.url + "/api/documents").json()
            if any(doc["id"] == expected_fixture_id for doc in documents):
                token = context.request.get(args.url + "/api/status").json()["token"]
                context.request.delete(args.url + f"/api/documents/{expected_fixture_id}", headers={"X-Workspace-Token": token})
            final_ids = {doc["id"] for doc in context.request.get(args.url + "/api/documents").json()}
            assert final_ids == original_ids, "Browser verification changed the original collection."

        mobile = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        mobile.route("**/*", route_request)
        mobile_page = mobile.new_page()
        mobile_page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        mobile_page.goto(args.url)
        expect(mobile_page.locator("#operation-banner")).to_be_hidden(timeout=30000)
        expect(mobile_page.locator("#preview-image")).to_be_visible(timeout=30000)
        no_overflow(mobile_page)
        mobile_page.screenshot(path=str(output / "mobile-library.png"), full_page=True)
        mobile_page.locator("#menu-button").click()
        expect(mobile_page.locator("#sidebar")).to_have_class("sidebar open")
        mobile_page.locator("#nav-workspace").click()
        expect(mobile_page.locator("#drawer-scrim")).to_be_hidden()
        mobile_page.locator("#question").fill("What does correlation mean?")
        mobile_page.locator("#run-query").click()
        expect(mobile_page.locator(".source-item").first).to_be_visible(timeout=60000)
        expect(mobile_page.locator("#operation-banner")).to_be_hidden(timeout=10000)
        no_overflow(mobile_page)
        mobile_page.screenshot(path=str(output / "mobile-search.png"), full_page=True)
        report["screenshots"] += ["mobile-library.png", "mobile-search.png"]
        checked("mobile navigation, search, previews, and no horizontal overflow")
        assert not report["external_requests"], report["external_requests"]
        assert not report["page_errors"], report["page_errors"]
        checked("no external browser requests or JavaScript errors")
        mobile.close(); context.close(); browser.close()
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
