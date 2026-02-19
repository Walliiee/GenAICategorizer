"""Playwright-based UI tests for the GenAI Categorizer web dashboard.

These tests start the FastAPI server, open it in a headless browser, and
verify the key UI flows: page load, file upload, dashboard rendering,
table interactions, and the reset flow.

Run locally::

    pip install -e ".[ui]"
    playwright install chromium
    pytest tests/test_ui.py -v

Requires the ``live_server_url`` and ``sample_json_file`` fixtures from conftest.
"""

import re

import pytest

pw = pytest.importorskip("playwright", reason="pip install -e '.[ui]' to run UI tests")
from playwright.sync_api import Page, expect  # noqa: E402

pytestmark = pytest.mark.ui


# ---------------------------------------------------------------------------
# Page load & structure
# ---------------------------------------------------------------------------


class TestPageLoad:
    """Verify the dashboard loads with the correct structure."""

    def test_title(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        expect(page).to_have_title("GenAI Categorizer")

    def test_header_visible(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        expect(page.locator("header")).to_be_visible()
        expect(page.locator(".logo h1")).to_have_text("GenAI Categorizer")

    def test_upload_zone_visible(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        drop_zone = page.locator("#drop-zone")
        expect(drop_zone).to_be_visible()
        expect(drop_zone.locator("h2")).to_have_text("Drop your conversation files here")

    def test_dashboard_hidden_initially(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        expect(page.locator("#dashboard")).to_be_hidden()

    def test_browse_button_present(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        expect(page.locator(".browse-btn")).to_be_visible()
        expect(page.locator(".browse-btn")).to_have_text("Browse Files")

    def test_file_input_accepts_json_csv(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        file_input = page.locator("#file-input")
        expect(file_input).to_have_attribute("accept", ".json,.csv")


# ---------------------------------------------------------------------------
# File upload & analysis
# ---------------------------------------------------------------------------


class TestFileUploadAndAnalysis:
    """Test uploading files and viewing the resulting dashboard."""

    def test_upload_json_shows_dashboard(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)

        page.locator("#file-input").set_input_files(str(sample_json_file))

        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)
        expect(page.locator("#dashboard")).to_be_visible()
        expect(page.locator("#upload-section")).to_be_hidden()

    def test_summary_cards_populated(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))
        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)

        total_el = page.locator("#stat-total")
        expect(total_el).not_to_have_text("0")

    def test_charts_rendered(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))
        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)

        for canvas_id in [
            "chart-categories",
            "chart-languages",
            "chart-complexity",
            "chart-voice",
            "chart-confidence",
        ]:
            expect(page.locator(f"#{canvas_id}")).to_be_visible()

    def test_table_has_rows(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))
        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)

        rows = page.locator("#table-body tr")
        expect(rows.first).to_be_visible()
        assert rows.count() >= 1

    def test_new_analysis_button_appears(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))
        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)

        expect(page.locator("#btn-new")).to_be_visible()

    def test_success_toast_appears(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))

        toast = page.locator(".toast.success")
        expect(toast).to_be_visible(timeout=30_000)
        expect(toast).to_contain_text("categorized")


# ---------------------------------------------------------------------------
# Table interactions
# ---------------------------------------------------------------------------


class TestTableInteractions:
    """Test search, filter, sort, and pagination in the data table."""

    def _load_dashboard(self, page: Page, live_server_url: str, sample_json_file):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))
        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)

    def test_search_filters_rows(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        self._load_dashboard(page, live_server_url, sample_json_file)

        page.locator("#search-input").fill("Python")
        page.wait_for_timeout(300)

        rows = page.locator("#table-body tr")
        assert rows.count() >= 1
        expect(rows.first).to_contain_text(re.compile(r"python|Python|code|Code", re.IGNORECASE))

    def test_category_filter_dropdown(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        self._load_dashboard(page, live_server_url, sample_json_file)

        select = page.locator("#category-filter")
        options = select.locator("option")
        assert options.count() >= 2

    def test_column_sort(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        self._load_dashboard(page, live_server_url, sample_json_file)

        confidence_header = page.locator("thead th[data-col='confidence_score']")
        confidence_header.click()

        arrow = confidence_header.locator(".sort-arrow")
        expect(arrow).to_have_text("▲")

        confidence_header.click()
        expect(arrow).to_have_text("▼")

    def test_text_preview_expands(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        self._load_dashboard(page, live_server_url, sample_json_file)

        preview = page.locator(".text-preview").first
        expect(preview).to_be_visible()

        preview.click()
        expect(preview).to_have_class(re.compile(r"expanded"))

        preview.click()
        expect(preview).not_to_have_class(re.compile(r"expanded"))


# ---------------------------------------------------------------------------
# Reset flow
# ---------------------------------------------------------------------------


class TestResetFlow:
    """Test the 'New Analysis' reset functionality."""

    def test_reset_returns_to_upload(
        self, page: Page, live_server_url: str, sample_json_file
    ):
        page.goto(live_server_url)
        page.locator("#file-input").set_input_files(str(sample_json_file))
        page.wait_for_selector("#dashboard:not(.hidden)", timeout=30_000)

        page.locator("#btn-new").click()

        expect(page.locator("#upload-section")).to_be_visible()
        expect(page.locator("#dashboard")).to_be_hidden()
        expect(page.locator("#btn-new")).to_be_hidden()


# ---------------------------------------------------------------------------
# Accessibility basics
# ---------------------------------------------------------------------------


class TestAccessibility:
    """Basic accessibility checks."""

    def test_page_has_lang_attribute(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        expect(page.locator("html")).to_have_attribute("lang", "en")

    def test_file_input_is_hidden_but_functional(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        file_input = page.locator("#file-input")
        expect(file_input).to_have_attribute("type", "file")

    def test_viewport_meta_tag(self, page: Page, live_server_url: str):
        page.goto(live_server_url)
        viewport = page.locator("meta[name='viewport']")
        expect(viewport).to_have_attribute("content", re.compile(r"width=device-width"))
