"""
Sanctuary Computer Tool - Browser Automation
Playwright-based browser automation for Gemini Computer Use integration.

Playwright-based browser automation for Sanctuary.

Architecture:
- Playwright runs in a DEDICATED THREAD (solves Flask threading conflicts)
- Commands sent via thread-safe queue, results returned via queue
- Persistent browser context survives across chat turns and request threads
- Translates Gemini Computer Use function_calls into Playwright actions
- Returns screenshots after each action for the Gemini vision loop
"""

import base64
import logging
import threading
import queue
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Timeout for waiting on browser actions (seconds)
COMMAND_TIMEOUT = 30


class ComputerTool:
    """
    Browser automation via Playwright.

    Runs Playwright in a dedicated background thread to avoid Flask's
    threading conflicts. All browser operations are dispatched via a
    thread-safe command queue.
    """

    # Default viewport — Gemini maps its 1000x1000 grid onto this
    VIEWPORT_WIDTH = 1280
    VIEWPORT_HEIGHT = 800
    GEMINI_GRID = 1000

    def __init__(self):
        self._worker_thread: Optional[threading.Thread] = None
        self._cmd_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._running = False
        self._current_url = ""
        self._headless = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, headless: bool = False):
        """Launch browser in dedicated thread."""
        if self._running and self._worker_thread and self._worker_thread.is_alive():
            return

        # Clean up dead thread if needed
        if self._worker_thread and not self._worker_thread.is_alive():
            logger.warning("ComputerTool: Worker thread died, restarting")
            self._running = False

        self._headless = headless
        self._cmd_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._browser_worker,
            daemon=True,
            name="Sanctuary-Browser"
        )
        self._worker_thread.start()

        # Wait for startup confirmation
        try:
            result = self._result_queue.get(timeout=15)
            if result == "STARTED":
                self._running = True
                logger.info("ComputerTool: Browser launched in dedicated thread")
            else:
                logger.error(f"ComputerTool: Unexpected startup result: {result}")
        except queue.Empty:
            logger.error("ComputerTool: Browser startup timed out")

    def stop(self):
        """Shutdown browser thread."""
        if self._running:
            self._cmd_queue.put(None)  # Shutdown signal
            if self._worker_thread:
                self._worker_thread.join(timeout=5)
        self._running = False
        logger.info("ComputerTool: Browser stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._worker_thread and self._worker_thread.is_alive()

    @property
    def current_url(self) -> str:
        return self._current_url

    # ------------------------------------------------------------------
    # Browser worker thread — owns all Playwright objects
    # ------------------------------------------------------------------

    def _browser_worker(self):
        """
        Runs in dedicated thread. Owns Playwright, browser, context, page.
        Receives commands via queue, executes them, returns results.
        """
        from playwright.sync_api import sync_playwright

        pw = None
        browser = None
        context = None
        page = None

        try:
            pw = sync_playwright().start()
            browser = pw.chromium.launch(headless=self._headless)
            context = browser.new_context(
                viewport={"width": self.VIEWPORT_WIDTH, "height": self.VIEWPORT_HEIGHT},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()
            self._result_queue.put("STARTED")
        except Exception as e:
            logger.error(f"ComputerTool worker startup failed: {e}")
            self._result_queue.put(f"FAILED: {e}")
            return

        # Command loop — runs until shutdown signal (None)
        while True:
            try:
                cmd = self._cmd_queue.get()
                if cmd is None:
                    break  # Shutdown

                action_name, action_args = cmd
                result = self._execute_in_thread(page, action_name, action_args)
                self._result_queue.put(result)

            except Exception as e:
                logger.error(f"ComputerTool worker error: {e}")
                self._result_queue.put({
                    "success": False,
                    "error": str(e),
                    "screenshot": "",
                    "url": page.url if page else "",
                })

        # Cleanup
        try:
            if context:
                context.close()
            if browser:
                browser.close()
            if pw:
                pw.stop()
        except Exception as e:
            logger.warning(f"ComputerTool cleanup error: {e}")

        logger.info("ComputerTool worker thread exiting")

    def _execute_in_thread(self, page, action_name: str, action_args: dict) -> dict:
        """Execute a single action on the page. Runs INSIDE the worker thread."""

        # Special: accessibility snapshot returns text, not screenshot
        if action_name == "_accessibility":
            return {
                "success": True,
                "accessibility": self._get_accessibility(page),
                "url": page.url,
                "error": None,
            }

        handler = self._ACTION_MAP.get(action_name)
        if not handler:
            return {
                "success": False,
                "error": f"Unknown action: {action_name}",
                "screenshot": self._take_screenshot(page),
                "url": page.url,
            }

        try:
            handler(self, page, action_args)
            page.wait_for_timeout(500)
            screenshot = self._take_screenshot(page)
            return {
                "success": True,
                "screenshot": screenshot,
                "url": page.url,
                "error": None,
            }
        except Exception as e:
            logger.error(f"ComputerTool action '{action_name}' failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "screenshot": self._take_screenshot(page),
                "url": page.url,
            }

    # ------------------------------------------------------------------
    # Public methods — called from Flask request threads
    # ------------------------------------------------------------------

    def execute(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a Gemini Computer Use function_call.
        Thread-safe — sends command to worker thread and waits for result.
        """
        if not self.is_running:
            self.start(headless=self._headless)

        name = function_call.get("name", "")
        args = function_call.get("args", {})

        self._cmd_queue.put((name, args))

        try:
            result = self._result_queue.get(timeout=COMMAND_TIMEOUT)
            self._current_url = result.get("url", self._current_url)
            return result
        except queue.Empty:
            return {
                "success": False,
                "error": f"Action '{name}' timed out after {COMMAND_TIMEOUT}s",
                "screenshot": "",
                "url": self._current_url,
            }

    def screenshot(self) -> str:
        """Get current screenshot. Thread-safe."""
        result = self.execute({"name": "_screenshot", "args": {}})
        return result.get("screenshot", "")

    def accessibility_snapshot(self) -> str:
        """Get page structure as text. Thread-safe."""
        result = self.execute({"name": "_accessibility", "args": {}})
        return result.get("accessibility", "")

    # ------------------------------------------------------------------
    # Coordinate mapping
    # ------------------------------------------------------------------

    def _to_viewport(self, gx: float, gy: float) -> Tuple[int, int]:
        """Convert Gemini 1000x1000 grid coords to viewport pixels."""
        x = int((gx / self.GEMINI_GRID) * self.VIEWPORT_WIDTH)
        y = int((gy / self.GEMINI_GRID) * self.VIEWPORT_HEIGHT)
        return x, y

    # ------------------------------------------------------------------
    # Screenshot/accessibility helpers (run inside worker thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _take_screenshot(page) -> str:
        """JPEG screenshot as base64. Runs in worker thread."""
        try:
            jpeg_bytes = page.screenshot(type="jpeg", quality=60)
            return base64.b64encode(jpeg_bytes).decode("utf-8")
        except Exception:
            return ""

    def _do_screenshot(self, page, args: dict):
        """No-op action — screenshot is captured after every action."""
        pass

    @staticmethod
    def _get_accessibility(page) -> str:
        """
        Extract page structure via JavaScript. Runs in worker thread.
        Replaces the removed page.accessibility.snapshot() API (Playwright 1.58+).
        Returns a text summary of interactive elements and page content.
        """
        try:
            result = page.evaluate("""() => {
                const items = [];

                // Collect interactive elements
                const selectors = 'a, button, input, textarea, select, [role="button"], [role="link"], [role="tab"], [role="menuitem"], [onclick]';
                document.querySelectorAll(selectors).forEach((el, i) => {
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role') || tag;
                    const text = (el.textContent || '').trim().slice(0, 100);
                    const href = el.getAttribute('href') || '';
                    const type = el.getAttribute('type') || '';
                    const placeholder = el.getAttribute('placeholder') || '';
                    const ariaLabel = el.getAttribute('aria-label') || '';
                    const value = el.value || '';
                    const rect = el.getBoundingClientRect();

                    // Skip invisible elements
                    if (rect.width === 0 && rect.height === 0) return;

                    let desc = `[${i}] ${role}`;
                    if (ariaLabel) desc += ` "${ariaLabel}"`;
                    else if (text) desc += ` "${text}"`;
                    if (href) desc += ` href=${href.slice(0, 80)}`;
                    if (type) desc += ` type=${type}`;
                    if (placeholder) desc += ` placeholder="${placeholder}"`;
                    if (value) desc += ` value="${value.slice(0, 50)}"`;
                    desc += ` @(${Math.round(rect.x)},${Math.round(rect.y)})`;

                    items.push(desc);
                });

                // Get headings for page structure
                const headings = [];
                document.querySelectorAll('h1, h2, h3').forEach(h => {
                    const text = (h.textContent || '').trim().slice(0, 100);
                    if (text) headings.push(`${h.tagName}: ${text}`);
                });

                // Get page text summary
                const bodyText = (document.body?.innerText || '').slice(0, 2000);

                let output = `URL: ${window.location.href}\\n`;
                output += `Title: ${document.title}\\n\\n`;
                if (headings.length) output += `HEADINGS:\\n${headings.join('\\n')}\\n\\n`;
                output += `INTERACTIVE ELEMENTS (${items.length}):\\n${items.join('\\n')}\\n\\n`;
                output += `PAGE TEXT (truncated):\\n${bodyText}`;

                return output;
            }""")
            return result
        except Exception as e:
            logger.error(f"ComputerTool accessibility extraction failed: {e}")
            return f"Error extracting page structure: {e}"

    # ------------------------------------------------------------------
    # Action handlers — run INSIDE the worker thread
    # All take (self, page, args) instead of (self, args)
    # ------------------------------------------------------------------

    def _navigate(self, page, args: dict):
        url = args.get("url", "about:blank")
        page.goto(url, wait_until="domcontentloaded")

    def _click_at(self, page, args: dict):
        x, y = self._to_viewport(args.get("x", 0), args.get("y", 0))
        page.mouse.click(x, y)

    def _hover_at(self, page, args: dict):
        x, y = self._to_viewport(args.get("x", 0), args.get("y", 0))
        page.mouse.move(x, y)

    def _type_text_at(self, page, args: dict):
        x, y = self._to_viewport(args.get("x", 0), args.get("y", 0))
        text = args.get("text", "")
        clear = args.get("clear_before_typing", False)
        press_enter = args.get("press_enter_after", False)

        page.mouse.click(x, y)

        if clear:
            page.keyboard.press("Control+a")
            page.keyboard.press("Delete")

        page.keyboard.type(text, delay=50)

        if press_enter:
            page.keyboard.press("Enter")

    def _key_combination(self, page, args: dict):
        keys = args.get("keys", [])
        combo = "+".join(keys)
        page.keyboard.press(combo)

    def _scroll_document(self, page, args: dict):
        direction = args.get("direction", "down")
        amount = args.get("amount", 3)
        delta = amount * 100
        if direction == "up":
            delta = -delta
        page.mouse.wheel(0, delta)

    def _scroll_at(self, page, args: dict):
        x, y = self._to_viewport(args.get("x", 0), args.get("y", 0))
        direction = args.get("direction", "down")
        amount = args.get("amount", 3)
        delta = amount * 100
        if direction == "up":
            delta = -delta
        page.mouse.move(x, y)
        page.mouse.wheel(0, delta)

    def _drag_and_drop(self, page, args: dict):
        sx, sy = self._to_viewport(args.get("start_x", 0), args.get("start_y", 0))
        ex, ey = self._to_viewport(args.get("end_x", 0), args.get("end_y", 0))
        page.mouse.move(sx, sy)
        page.mouse.down()
        page.mouse.move(ex, ey, steps=10)
        page.mouse.up()

    def _go_back(self, page, args: dict):
        page.go_back()

    def _go_forward(self, page, args: dict):
        page.go_forward()

    def _wait(self, page, args: dict):
        page.wait_for_timeout(5000)

    def _open_browser(self, page, args: dict):
        url = args.get("url", "about:blank")
        page.goto(url, wait_until="domcontentloaded")

    def _search(self, page, args: dict):
        query = args.get("query", "")
        page.goto(
            f"https://www.google.com/search?q={query}",
            wait_until="domcontentloaded",
        )

    # Handler dispatch table
    _ACTION_MAP = {
        "navigate": _navigate,
        "click_at": _click_at,
        "hover_at": _hover_at,
        "type_text_at": _type_text_at,
        "key_combination": _key_combination,
        "scroll_document": _scroll_document,
        "scroll_at": _scroll_at,
        "drag_and_drop": _drag_and_drop,
        "go_back": _go_back,
        "go_forward": _go_forward,
        "wait_5_seconds": _wait,
        "open_web_browser": _open_browser,
        "search": _search,
        "_screenshot": _do_screenshot,
    }
