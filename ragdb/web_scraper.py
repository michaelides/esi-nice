import os
import asyncio
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from typing import List, Dict, Any # Added Any for failed_urls_details

def url_to_filename(url: str, max_length: int = 200) -> str:
    """Converts a URL to a safe filename for storing markdown, truncating if necessary."""
    if not url: # Handle empty URL input
        print("ERROR: web_scraper.url_to_filename: URL cannot be empty.")
        return "invalid_url.md"

    try:
        parsed_url = urlparse(url)
        # Ensure netloc and path are strings, even if empty
        netloc_str = parsed_url.netloc or ""
        path_str = parsed_url.path.strip('/') or ""

        filename_base = "_".join(filter(None, [netloc_str, path_str]))

        filename_base = "".join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in filename_base)

        while "__" in filename_base:
            filename_base = filename_base.replace("__", "_")

        if len(filename_base) > max_length - 3: # -3 for ".md"
            filename_base = filename_base[:max_length - 3]

        filename_base = filename_base.strip('_')

        if not filename_base:
            filename_base = netloc_str.replace('.', '_').replace(':', '_') if netloc_str else "scraped_page"
            if not filename_base: # Ultimate fallback if netloc was also empty or just dots/colons
                 filename_base = "default_scraped_page_name"
            if len(filename_base) > max_length -3 :
                filename_base = filename_base[:max_length -3]
            filename_base = filename_base.strip('_')
            if not filename_base: # If stripping underscores makes it empty
                filename_base = "final_fallback_name"


        return f"{filename_base}.md"
    except Exception as e:
        print(f"ERROR: web_scraper.url_to_filename: Failed to convert URL '{url}' to filename. Error: {e}")
        # Create a fallback filename based on a simple hash or uuid if parsing fails badly
        import hashlib
        return f"error_url_{hashlib.md5(url.encode()).hexdigest()[:10]}.md"


async def scrape_websites(urls: List[str], output_dir: str):
    """Scrapes a list of URLs and saves the content as markdown files."""
    print(f"LOG: web_scraper.Running web scraping for {len(urls)} URLs. Output to: {output_dir}")

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"ERROR: web_scraper.scrape_websites: Could not create output directory {output_dir}. Error: {e}")
        return # Cannot proceed

    if not urls:
        print("LOG: web_scraper.No URLs provided for scraping.")
        return

    crawler = None # Initialize crawler to None for finally block
    try:
        crawler = AsyncWebCrawler(should_markdown=True)
        print(f"LOG: web_scraper.Crawler initialized for {len(urls)} seed URLs.")
    except Exception as e:
        print(f"ERROR: web_scraper.scrape_websites: Error initializing AsyncWebCrawler: {e}")
        return

    saved_count = 0
    failed_urls_details: List[Dict[str, Any]] = [] # Type hint for clarity
    MAX_DEPTH = 1
    MAX_PAGES_PER_SEED = 10
    INCLUDE_EXTERNAL = False

    try:
        for start_url in urls:
            if not start_url or not isinstance(start_url, str):
                print(f"WARNING: web_scraper.scrape_websites: Invalid URL provided: {start_url}. Skipping.")
                failed_urls_details.append({"url": str(start_url), "error": "Invalid or empty URL string."})
                continue

            print(f"LOG: web_scraper.  Initiating deep crawl from: {start_url}")

            current_crawl_config = CrawlerRunConfig(
                deep_crawl_strategy=BFSDeepCrawlStrategy(
                    max_depth=MAX_DEPTH, include_external=INCLUDE_EXTERNAL, max_pages=MAX_PAGES_PER_SEED
                ),
                scraping_strategy=LXMLWebScrapingStrategy(),
                stream=True, verbose=True
            )

            try:
                async for result in await crawler.arun(url=start_url, config=current_crawl_config):
                    current_page_url = result.url if result else "Unknown URL (result was None)"
                    print(f"LOG: web_scraper.    Processing page: {current_page_url} (Origin: {start_url})")

                    if not result: # Handle cases where result might be None
                        print(f"WARNING: web_scraper.scrape_websites: Received None result for a page from {start_url}.")
                        failed_urls_details.append({"url": current_page_url, "error": "Crawler returned None result."})
                        continue

                    try:
                        if result.success and result.markdown:
                            filename = url_to_filename(current_page_url)
                            filepath = os.path.join(output_dir, filename)
                            try:
                                with open(filepath, "w", encoding="utf-8") as f:
                                    f.write(result.markdown)
                                print(f"LOG: web_scraper.      Successfully saved: {current_page_url} to {filepath}")
                                saved_count += 1
                            except (IOError, OSError) as e: # More specific file write errors
                                print(f"ERROR: web_scraper.scrape_websites: Error writing file {filepath} for URL {current_page_url}: {e}")
                                failed_urls_details.append({"url": current_page_url, "error": f"File write error: {e}"})
                        elif result.success and not result.markdown:
                            print(f"LOG: web_scraper.      Processed URL: {current_page_url}, but no markdown content extracted.")
                        else: # result.success is False or markdown is missing
                            error_message = result.error or "Unknown error during page crawl (no markdown and not success)."
                            print(f"WARNING: web_scraper.scrape_websites: Failed to scrape page {current_page_url}: {error_message}")
                            failed_urls_details.append({"url": current_page_url, "error": error_message})
                    except Exception as page_processing_e:
                        print(f"ERROR: web_scraper.scrape_websites: Error processing result for {current_page_url}: {page_processing_e}")
                        failed_urls_details.append({"url": current_page_url, "error": f"Result processing error: {page_processing_e}"})

            except Exception as e: # Errors during arun() for a specific start_url
                print(f"ERROR: web_scraper.scrape_websites: Error during crawl for start URL {start_url}: {e}")
                failed_urls_details.append({"url": start_url, "error": f"Crawl execution error: {str(e)}"})
    finally:
        if crawler and hasattr(crawler, 'close'):
            try:
                print("LOG: web_scraper.Closing crawler session...")
                await crawler.close()
            except Exception as e:
                print(f"ERROR: web_scraper.scrape_websites: Error closing crawler session: {e}")

    print(f"LOG: web_scraper.Deep scraping finished. Successfully saved {saved_count} pages from {len(urls)} seed URLs.")
    if failed_urls_details:
        print("WARNING: web_scraper.scrape_websites: Details for failed URLs/pages:")
        for detail in failed_urls_details:
            print(f"  URL: {detail['url']}, Error: {detail['error']}")
    print("LOG: web_scraper.Finished web scraping process.")

print("DEBUG: ragdb/web_scraper.py processed with error handling improvements.")
