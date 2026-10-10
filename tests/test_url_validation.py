import os
import sys
import unittest

# Ensure colibri root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entrypoint import (
    extract_and_clean_url,
    validate_posting_url,
    canonical_url,
    seen_before,
    record_posting,
    _format_search_results,
    _extract_job_links,
)


class TestPostingUrlValidation(unittest.TestCase):
    def test_extract_and_clean_url(self):
        # Markdown link
        raw_md = "[Senior Backend Engineer](https://jobs.lever.co/company/abc-123)"
        self.assertEqual(extract_and_clean_url(raw_md), "https://jobs.lever.co/company/abc-123")

        # Quoted string
        raw_quoted = '"https://job-boards.greenhouse.io/org/jobs/456"'
        self.assertEqual(extract_and_clean_url(raw_quoted), "https://job-boards.greenhouse.io/org/jobs/456")

        # Angle brackets
        raw_bracket = "<https://jobs.ashbyhq.com/org/789>"
        self.assertEqual(extract_and_clean_url(raw_bracket), "https://jobs.ashbyhq.com/org/789")

    def test_reject_search_and_directory_urls(self):
        bad_urls = [
            "site:jobs.ashbyhq.com senior backend engineer remote",
            "site:boards.greenhouse.io",
            "https://boards.greenhouse.io",
            "https://boards.greenhouse.io/company",
            "site:jobs.lever.co",
            "https://jobs.lever.co",
            "https://jobs.lever.co/company",
            "https://jobs.ashbyhq.com",
            "https://jobs.ashbyhq.com/company",
            "https://glassdoor.com",
            "https://www.glassdoor.com/Job/seattle-jobs-SRCH_IL.0,7_IC1150505.htm",
            "https://indeed.com",
            "https://indeed.com/jobs?q=senior+backend+engineer+remote",
            "https://www.linkedin.com/jobs/search?keywords=backend",
            "https://www.ziprecruiter.com/jobs-search?search=backend",
            "https://amazon.jobs/en-US/jobs/...",
            "https://ashby.co.uk/jobs",
            "https://kentrenton.com/careers",
            "https://google.com/careers",
            "https://waymo.com/careers",
            "https://kotlinlang.org/careers",
            "https://truncated-search.com/jobs",
            "Backend Engineers design and operate...",
        ]
        for url in bad_urls:
            valid, reason = validate_posting_url(url)
            self.assertFalse(valid, f"Expected '{url}' to be rejected, but was accepted")
            self.assertTrue(len(reason) > 0)

    def test_accept_direct_posting_urls(self):
        good_urls = [
            "https://job-boards.greenhouse.io/surtai/jobs/4145680009",
            "https://boards.greenhouse.io/github/jobs/1234567",
            "https://jobs.lever.co/zerion/d10ab83b-f425-49c2-bc93-09a14e97c2b8",
            "https://jobs.lever.co/raya/675d586b-d156-45ed-9d13-ba1e8dcddd49",
            "https://jobs.ashbyhq.com/resend/9b3f3b9e-63f8-4e3a-9e32-23c5d6e279aa",
            "https://www.databricks.com/company/careers/engineering---pipeline/senior-software-engineer---distributed-data-systems-6936994002",
        ]
        for url in good_urls:
            valid, reason = validate_posting_url(url)
            self.assertTrue(valid, f"Expected '{url}' to be accepted, but was rejected: {reason}")

    def test_search_results_formatting(self):
        raw_markdown = """## Search Results

[Raya - Senior Backend Engineer](https://jobs.lever.co/raya/675d586b-d156-45ed-9d13-ba1e8dcddd49)
Building scalable distributed backend services...

[Indeed - Search](https://www.indeed.com/jobs?q=senior+backend+engineer)
Find jobs in Seattle...
"""
        formatted = _format_search_results(raw_markdown, "senior backend engineer")
        self.assertIn("DIRECT JOB POSTING", formatted)
        self.assertIn("SEARCH/DIRECTORY PAGE", formatted)
        self.assertIn("URL: https://jobs.lever.co/raya/675d586b-d156-45ed-9d13-ba1e8dcddd49", formatted)
        self.assertIn("DO NOT record this URL", formatted)

    def test_record_posting_rejects_search_result_page(self):
        # Trying to record a search-result URL must return an Error
        res = record_posting(
            url="https://www.indeed.com/jobs?q=senior+backend",
            title="Senior Backend Engineer",
            company="Unknown",
            status="candidate",
            score=8,
            notes="Notes here",
        )
        self.assertTrue(res.startswith("Error:"), f"Expected Error response, got: {res}")
        self.assertIn("not an individual job posting URL", res)

    def test_seen_before_warns_on_search_result_page(self):
        res = seen_before("https://boards.greenhouse.io")
        self.assertTrue(res.startswith("Warning:"), f"Expected Warning response, got: {res}")

    def test_extract_job_links_from_directory_html(self):
        html = """
        <html>
            <body>
                <h1>Careers</h1>
                <a href="/jobs/search?q=eng">Search More</a>
                <a href="https://jobs.lever.co/acme/12345678-abcd-1234-abcd-1234567890ab">Senior Backend Engineer</a>
                <a href="https://job-boards.greenhouse.io/acme/jobs/9876543">Staff Systems Engineer</a>
            </body>
        </html>
        """
        links = _extract_job_links(html, "https://example.com/careers")
        urls = [url for _, url in links]
        self.assertIn("https://jobs.lever.co/acme/12345678-abcd-1234-abcd-1234567890ab", urls)
        self.assertIn("https://job-boards.greenhouse.io/acme/jobs/9876543", urls)
        self.assertNotIn("https://example.com/jobs/search?q=eng", urls)


if __name__ == "__main__":
    unittest.main()
