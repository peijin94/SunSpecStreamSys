const{chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const assert=require('node:assert/strict');
(async()=>{const browser=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE || undefined});try{
 const page=await browser.newPage();await page.goto(process.env.LWA_TEST_URL||'https://ovsa.njit.edu/live/');
 await page.waitForFunction(()=>+document.querySelector('[data-testid="spectrum-canvas-I"]')?.dataset.end>1000000000);
 await page.getByTestId('spectrum-pause').click();
 const end=Math.floor(+(await page.getByTestId('spectrum-canvas-I').getAttribute('data-end')))-120;
 await page.getByText('Select UTC time range',{exact:true}).click();
 await page.getByLabel('Range from UTC').fill(new Date((end-60)*1000).toISOString().slice(0,19));
 await page.getByLabel('Range to UTC').fill(new Date(end*1000).toISOString().slice(0,19));
 await page.route('**/spectrum-range?**',async route=>{await new Promise(resolve=>setTimeout(resolve,900));try{await route.continue();}catch{}});
 const first=page.waitForRequest(r=>r.url().includes('/spectrum-range?')&&new URL(r.url()).searchParams.has('start'));
 await page.getByRole('button',{name:'Apply range & pause'}).click();await first;
 const next=page.waitForRequest(r=>r.url().includes('/spectrum-range?')&&new URL(r.url()).searchParams.get('max_rows')==='600');
 await page.getByTestId('range-600').click();const selected=new URL((await next).url());
 assert.equal(Number(selected.searchParams.get('end')),end,'Paused preset must preserve newly selected UTC end while its prior request is pending');
 console.log('PASS paused preset preserves pending custom UTC selection end',end);
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
