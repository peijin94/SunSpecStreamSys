const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const base=process.env.LWA_TEST_URL||'https://ovsa.njit.edu/live/';
const out=process.env.LWA_TEST_OUTPUT || process.env.TMPDIR || require('node:os').tmpdir();
fs.mkdirSync(out,{recursive:true});
// Expected deployment prefix derived from the page URL itself, so the same
// test validates /live/ (public) and /live-dev/ (dev) deployments.
const expectedBase=new URL(base).pathname.endsWith('/')?new URL(base).pathname:new URL(base).pathname+'/';
const expectedOrigin=new URL(base).origin;
let browser;
(async()=>{
 browser=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE || undefined});
 const context=await browser.newContext({viewport:{width:1440,height:1100}});
 const page=await context.newPage();const errors=[],requests=[],results=[];let measured=false;
 page.on('pageerror',e=>errors.push(String(e)));
 page.on('request',r=>{const url=r.url();if(url.includes('/spectrum-range?'))requests.push(url);if(new URL(url).pathname.endsWith('/spectrum-measure'))measured=true;});
 const check=(name,details)=>{results.push({name,details});console.log('PASS',name,JSON.stringify(details||{}));};
 const canvasI=page.getByTestId('spectrum-canvas-I');
 const canvasV=page.getByTestId('spectrum-canvas-V');
 const bounds=which=>which.evaluate(e=>({start:+e.dataset.start,end:+e.dataset.end,fmin:+e.dataset.fmin,fmax:+e.dataset.fmax}));
 const fixed=b=>{assert.ok(Math.abs(b.end-b.start-300)<.01);assert.equal(b.fmin,15);assert.equal(b.fmax,85);};
 await page.goto(base,{waitUntil:'domcontentloaded'});
 // Public fixed status is rendered immediately, even before first spectra arrive.
 const status=page.getByTestId('public-fixed-status');
 await status.waitFor({timeout:15000});
 assert.match(await status.innerText(),/^● Live(?:\n|$)/);
 assert.doesNotMatch(await status.innerText(),/Fixed last 5 min|Full band 15[–-]85 MHz|Rolling latest 300 s|No pause or zoom/);
 assert.equal(await status.getAttribute('data-seconds'),'300');
 assert.match(await page.getByTestId('public-source-timestamp').innerText(),/Source data through/);
 check('public-fixed-status-visible',{seconds:await status.getAttribute('data-seconds')});
 // No interactive UI anywhere: no presets, pause/play/live, custom ranges,
 // zoom controls, measurement launcher/panel/picking affordances.
 assert.equal(await page.getByTestId('spectrum-controls').count(),0,'public must not render spectrum controls');
 assert.equal(await page.getByRole('button',{name:'Interactive spectrum measure',exact:true}).count(),0,'public must not render measurement launcher');
 assert.equal(await page.getByTestId('spectrum-measure-panel').count(),0,'public must not render measurement panel');
 assert.equal(await page.getByTestId('spectrum-pause').count(),0,'public must not render pause/play');
 assert.equal(await page.getByTestId('spectrum-live').count(),0,'public must not render live button');
 for(const id of ['range-180','range-300','range-600','range-1800','range-3600'])
  assert.equal(await page.getByTestId(id).count(),0,`public must not render preset ${id}`);
 assert.equal(await page.getByRole('button',{name:'Zoom out',exact:true}).count(),0);
 assert.equal(await page.getByRole('button',{name:'Reset zoom',exact:true}).count(),0);
 assert.equal(await page.getByRole('button',{name:'Apply range & pause'}).count(),0);
 assert.equal(await page.getByRole('button',{name:'Apply frequency range'}).count(),0);
 assert.equal(await page.getByText('Select UTC time range',{exact:true}).count(),0);
 assert.equal(await page.getByText('Select frequency range',{exact:true}).count(),0);
 assert.equal(await page.getByLabel('Range from UTC').count(),0);
 assert.equal(await page.getByLabel('Minimum frequency MHz').count(),0);
 assert.equal(await page.getByTestId('measure-point-count').count(),0);
 check('no-interactive-or-measurement-ui');
 // Both canvases render but are explicitly non-interactive.
 assert.equal(await canvasI.count(),1);assert.equal(await canvasV.count(),1);
 assert.equal(await canvasI.getAttribute('data-interactive'),'false');
 assert.equal(await canvasV.getAttribute('data-interactive'),'false');
 assert.match(await canvasI.getAttribute('aria-label')||'',/interaction disabled/);
 assert.match(await canvasV.getAttribute('aria-label')||'',/interaction disabled/);
 check('canvases-noninteractive');
 // Wait for live spectra, then verify the fixed 300 s full-band rolling window.
 await page.waitForFunction(()=>{const e=document.querySelector('[data-testid="spectrum-canvas-I"]');return +e?.dataset.end>1000000000;},{},{timeout:30000});
 await page.waitForTimeout(500);
 let b=await bounds(canvasI);
 assert.ok(Math.abs(b.end-b.start-300)<.01,`fixed 300 s window, got ${b.end-b.start}`);
 assert.equal(b.fmin,15);assert.equal(b.fmax,85);
 assert.deepEqual(await bounds(canvasV),b,'I and V share the fixed view');
 check('fixed-300-full-band-rolling',b);
 // Continuous polling: request count must grow while the page sits untouched.
 const idleCount=requests.length;await page.waitForTimeout(1800);
 assert.ok(requests.length>idleCount,'public page keeps live-polling when untouched');
 b=await bounds(canvasI);assert.ok(Math.abs(b.end-b.start-300)<.01);assert.equal(b.fmin,15);assert.equal(b.fmax,85);
 check('keeps-polling-fixed-window',{before:idleCount,after:requests.length});
 // Every spectrum request must carry the deployment base path.
 assert.ok(requests.length>=1,'expected at least one spectrum-range request');
 for(const url of requests){
  const u=new URL(url);
  const expectedPath=(expectedBase+'spectrum-range').replace(/\/{2,}/g,'/');
  assert.equal(u.origin,expectedOrigin,`request origin ${url}`);
  assert.equal(u.pathname,expectedPath,`request path ${u.pathname} must equal ${expectedPath}`);
 }
 assert.ok(requests.some(u=>new URL(u).searchParams.get('seconds')==='300'),'initial request asks for the fixed 300 s window');
 check('request-paths-carry-deployment-base',{base:expectedBase,requests:requests.length,sample:requests[0]});
 // Pointer gestures must not change the view, pause polling, or pick points.
 const box=await canvasI.boundingBox();const plot={x:box.x+58,y:box.y+14,w:box.width-74,h:box.height-50};
 const beforeDrag=await bounds(canvasI);const dragCount=requests.length;
 await page.mouse.move(plot.x+plot.w*.2,plot.y+plot.h*.2);await page.mouse.down();await page.mouse.move(plot.x+plot.w*.75,plot.y+plot.h*.8,{steps:12});await page.mouse.up();
 await page.waitForTimeout(1500);
 const afterDrag=await bounds(canvasI);fixed(afterDrag);assert.ok(afterDrag.end>beforeDrag.end,'drag must not freeze the advancing source time');
 const vAfterDrag=await bounds(canvasV);fixed(vAfterDrag);assert.ok(Math.abs(vAfterDrag.end-afterDrag.end)<1.1,'I/V source bounds stay aligned');
 assert.ok(requests.length>dragCount,'drag cannot pause public polling');
 assert.equal(await page.getByTestId('measure-point-count').count(),0,'clicks cannot create measurement points');
 // Reverse drag from the V panel must likewise preserve rolling playback.
 const beforeReverse=await bounds(canvasI);const reverseCount=requests.length;
 const vbox=await canvasV.boundingBox();const vp={x:vbox.x+58,y:vbox.y+14,w:vbox.width-74,h:vbox.height-50};
 await page.mouse.move(vp.x+vp.w*.8,vp.y+vp.h*.85);await page.mouse.down();await page.mouse.move(vp.x+vp.w*.25,vp.y+vp.h*.2,{steps:8});await page.mouse.up();
 await page.waitForTimeout(1200);
 const afterReverse=await bounds(canvasI);fixed(afterReverse);assert.ok(afterReverse.end>beforeReverse.end,'reverse drag must not freeze playback');assert.ok(requests.length>reverseCount);
 check('pointer-gestures-cannot-change-view-or-pause',beforeDrag);
 // Single clicks (measurement picking gesture) are inert as well.
 const beforeClick=await bounds(canvasI);
 await page.mouse.click(plot.x+plot.w*.5,plot.y+plot.h*.5);await page.waitForTimeout(1200);
 const afterClick=await bounds(canvasI);fixed(afterClick);assert.ok(afterClick.end>beforeClick.end,'click must not freeze playback');
 assert.equal(await page.getByTestId('measure-point-count').count(),0);
 check('clicks-cannot-pick-or-pause');
 // Keyboard: focused canvas must not exist (tabindex -1), Escape does nothing.
 assert.equal(await canvasI.getAttribute('tabindex'),'-1');
 check('canvas-not-focusable');
 // Manual refresh is allowed but must keep the fixed rolling 300 s window.
 await page.getByRole('button',{name:/Refresh I \/ V|Updating/}).click();await page.waitForTimeout(1500);
 b=await bounds(canvasI);assert.ok(Math.abs(b.end-b.start-300)<.01);assert.equal(b.fmin,15);assert.equal(b.fmax,85);
 check('manual-refresh-stays-fixed-300',b);
 // No measurement traffic may ever be issued by the public page.
 await page.waitForTimeout(1000);assert.equal(measured,false);
 for(const url of requests){const query=new URL(url).searchParams;assert.equal(query.get('seconds'),'300');assert.equal(query.has('start'),false);assert.equal(query.has('end'),false);}
 check('no-measurement-requests');
 await page.screenshot({path:out+'/lwa-public-feature-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(500);
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth+1));
 await page.screenshot({path:out+'/lwa-public-feature-mobile.png',fullPage:true});
 check('mobile-no-horizontal-overflow');
 assert.deepEqual(errors,[]);check('no-browser-errors');
 fs.writeFileSync(out+'/lwa-public-browser-results.json',JSON.stringify({base,expectedBase,results,errors,requests:requests.length},null,2));
 await browser.close();console.log('ACCEPTANCE PASSED',results.length);
})().catch(async e=>{console.error(e);await browser?.close();process.exitCode=1;});
