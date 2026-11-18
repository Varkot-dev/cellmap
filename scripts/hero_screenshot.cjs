// Renders the README's opening image from the real atlas page.
//
// Usage (needs Node and Playwright with Chromium):
//     node scripts/hero_screenshot.cjs        (with Playwright installed, globally or locally)
//
// Opens site/atlas.html, colours the cells by cell type, rotates to a fixed angle, and saves
// assets/hero_atlas.png. The picture is the page itself, not a separate drawing.
const { chromium } = require('playwright');
const path = require('node:path');

(async () => {
const root = path.resolve(__dirname, '..');
const browser = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 1400, height: 820 }, deviceScaleFactor: 2 });
await page.goto('file://' + path.join(root, 'site', 'atlas.html'));
await page.waitForTimeout(1500);
await page.selectOption('#color-by', 'coarse');
await page.click('#methods button:nth-child(4)'); // the hand-written VAE
await page.waitForTimeout(2200);
await page.screenshot({ path: path.join(root, 'assets', 'hero_atlas.png') });
await browser.close();
console.log('wrote assets/hero_atlas.png');
})();
