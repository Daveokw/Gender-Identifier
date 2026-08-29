const { chromium } = require('playwright');

(async () => {
  const url = process.argv[2];
  if (!url) {
    console.error("Error: Please provide a URL as an argument.");
    process.exit(1);
  }

  console.log(`Starting keep-alive check for: ${url}`);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
  });
  const page = await context.newPage();

  try {
    console.log("Navigating to the app URL...");
    // Streamlit apps can be slow to initialize, so we use a generous timeout
    await page.goto(url, { waitUntil: 'networkidle', timeout: 90000 });
  } catch (err) {
    console.log("Note: Page load timed out or encountered an error, checking for sleep indicators anyway:", err.message);
  }

  // Wait a bit to ensure the sleep screen has fully rendered if present
  await page.waitForTimeout(10000);

  try {
    // Check if the "App is sleeping" overlay exists
    const sleepingOverlay = await page.locator('text="Yes, get this app back up!"');
    if (await sleepingOverlay.count() > 0) {
      console.log("App is sleeping! Attempting to wake it up...");
      await sleepingOverlay.first().click();
      console.log("Clicked the wake up button.");
      
      // Wait for it to wake up (Streamlit says it can take a few minutes)
      console.log("Waiting for the app to wake up (this may take a few minutes)...");
      await page.waitForTimeout(60000);
      
      // Take a screenshot to verify
      await page.screenshot({ path: 'wakeup_screenshot.png' });
      console.log("Screenshot saved as wakeup_screenshot.png");
    } else {
      console.log("App appears to be awake already. No action needed.");
    }
  } catch (err) {
    console.log("Error while checking for sleep overlay:", err.message);
  }

  await browser.close();
  console.log("Keep-alive check completed.");
})();
