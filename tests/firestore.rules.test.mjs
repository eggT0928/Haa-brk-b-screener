import { readFile } from 'node:fs/promises';
import { after, before, beforeEach, test } from 'node:test';
import { initializeTestEnvironment, assertFails, assertSucceeds } from '@firebase/rules-unit-testing';
import { collection, deleteDoc, deleteField, doc, getDoc, getDocs, limit, orderBy, query, serverTimestamp, setDoc, updateDoc } from 'firebase/firestore';

let env;
const tickers = ['SPY','SPYM','VEA','VWO','IWM','BIL','IEF','TLT','TIP','PDBC','VNQ','BRK-B'];
const profile = () => ({holdings:Object.fromEntries(tickers.map(t=>[t,0])),cash:10000,sp500:'SPY',backtestStart:'',backtestEnd:'',updatedAt:serverTimestamp()});
before(async () => {
  env = await initializeTestEnvironment({projectId:'demo-haa',firestore:{rules:await readFile(new URL('../firestore.rules',import.meta.url),'utf8')}});
});
beforeEach(async () => {
  await env.clearFirestore();
  await env.withSecurityRulesDisabled(async ctx => {
    const db = ctx.firestore();
    await setDoc(doc(db,'access/owner'),{enabled:true});
    await setDoc(doc(db,'access/other'),{enabled:true});
    await setDoc(doc(db,'access/wife'),{enabled:true,role:'member',ownerUid:'owner'});
    await setDoc(doc(db,'signals/confirmed'),{month:'2026-07'});
    await setDoc(doc(db,'internal/history'),{payload:'비공개 캐시'});
  });
});
after(async () => { await env?.cleanup(); });
const owner = () => env.authenticatedContext('owner').firestore();
const sujinTickers=['QQQM','SPY','SPYM','JEPQ','BRK-B','IEF','TLT','GLD','PDBC'];
const sujinProfile=()=>({strategy:'sujin-fixed-v1',holdings:Object.fromEntries(sujinTickers.map(t=>[t,0])),totalMode:'manual',totalBalance:10000,cash:0,quantityMode:'fractional',backtestStart:'',backtestEnd:'',updatedAt:serverTimestamp()});
const sujinRecord=()=>({createdAt:serverTimestamp(),month:'2026-08',strategy:'sujin-fixed-v1',kind:'계산안',equity:10000,remainingCash:0,priceUpdatedAt:'2026-08-28T21:30:00Z',oldestPriceAt:'2026-08-28T20:00:00Z',quantityMode:'fractional',lines:sujinTickers.map(ticker=>({ticker,held:0,price:100,target:10,trade:10}))});
const pensionCodes=['379800','294400','283580','453810','411060','385560','308620','453850','449170'];
const pensionWeights=()=>Object.fromEntries(pensionCodes.map((code,i)=>[code,[.24,.08,.08,.08,.19,.14,.07,.07,.05][i]]));
const pensionProfile=()=>({strategy:'magic-pension-v1',currency:'KRW',totalBalance:1000000,principal:800000,
  holdings:Object.fromEntries(pensionCodes.map(code=>[code,0])),weights:pensionWeights(),
  targetOverrides:Object.fromEntries(pensionCodes.map(code=>[code,null])),updatedAt:serverTimestamp()});
const pensionRecord=()=>({createdAt:serverTimestamp(),month:'2026-10',strategy:'magic-pension-v1',currency:'KRW',kind:'계산안',equity:1000000,principal:800000,remainingCash:0,
  weights:pensionWeights(),priceUpdatedAt:'2026-10-26T06:30:00Z',oldestPriceAt:'2026-10-26T06:30:00Z',
  // 종목은 고정된 pensionCodes 순서로 결정하고 매매수량은 target-held로 복원한다.
  lines:pensionCodes.map(code=>({held:0,price:10000,target:Math.round(pensionWeights()[code]*100)}))});

test('미인증 사용자는 공용 신호를 읽을 수 없다', async () => {
  await assertFails(getDoc(doc(env.unauthenticatedContext().firestore(),'signals/confirmed')));
});
test('로그인했어도 미승인 사용자는 차단한다', async () => {
  await assertFails(getDoc(doc(env.authenticatedContext('stranger').firestore(),'signals/confirmed')));
});
test('승인 사용자는 공용 신호를 읽고 본인 설정을 저장한다', async () => {
  await assertSucceeds(getDoc(doc(owner(),'signals/confirmed')));
  await assertSucceeds(setDoc(doc(owner(),'users/owner'),profile()));
});
test('다른 사용자의 보유수량을 읽거나 쓰지 못한다', async () => {
  await assertFails(getDoc(doc(owner(),'users/other')));
  await assertFails(setDoc(doc(owner(),'users/other'),profile()));
});
test('신호, 가격, 서버 캐시와 승인 목록에 쓰지 못한다', async () => {
  for (const path of ['signals/confirmed','signals/preview','market/current','internal/history','access/owner','signalHistory/2026-07','status/daily']) {
    await assertFails(setDoc(doc(owner(),path),{enabled:true}));
  }
  await assertFails(getDoc(doc(owner(),'internal/history')));
});
test('음수·NaN·미지원 티커·타임스탬프 위조를 거부한다', async () => {
  const p = profile();
  await assertFails(setDoc(doc(owner(),'users/owner'),{...p,cash:-1}));
  await assertFails(setDoc(doc(owner(),'users/owner'),{...p,holdings:{...p.holdings,SPY:NaN}}));
  await assertFails(setDoc(doc(owner(),'users/owner'),{...p,holdings:{...p.holdings,FAKE:3}}));
  await assertFails(setDoc(doc(owner(),'users/owner'),{...p,sp500:'IVV'}));
  await assertFails(setDoc(doc(owner(),'users/owner'),{...p,updatedAt:'2026-01-01'}));
});
test('자기 승인 상태만 읽고 다른 승인 목록을 열람하지 못한다', async () => {
  await assertSucceeds(getDoc(doc(owner(),'access/owner')));
  await assertFails(getDoc(doc(owner(),'access/other')));
  await assertFails(getDocs(collection(owner(),'access')));
});
test('리밸런싱 계산안은 본인 경로에 추가만 가능하다', async () => {
  const data = {createdAt:serverTimestamp(),signalMonth:'2026-07',sp500:'SPYM',equity:10000,kind:'계산안',lines:[]};
  await assertSucceeds(setDoc(doc(owner(),'users/owner/rebalances/one'),data));
  await assertFails(updateDoc(doc(owner(),'users/owner/rebalances/one'),{equity:1}));
  await assertFails(deleteDoc(doc(owner(),'users/owner/rebalances/one')));
  await assertFails(getDoc(doc(env.authenticatedContext('other').firestore(),'users/owner/rebalances/one')));
});

test('수진 저장이 현근 보유수량을 덮어쓰지 않는다',async()=>{
  await assertSucceeds(setDoc(doc(owner(),'users/owner'),profile()));
  await assertSucceeds(setDoc(doc(owner(),'users/owner/portfolios/sujin'),sujinProfile()));
  const unchanged=await getDoc(doc(owner(),'users/owner'));
  if(unchanged.data().sp500!=='SPY'||unchanged.data().cash!==10000) throw new Error('현근 변경됨');
  await assertFails(setDoc(doc(owner(),'users/owner/portfolios/other'),sujinProfile()));
});
test('수진 데이터도 미인증·미승인·다른 계정 접근을 거부한다',async()=>{
  for(const db of [env.unauthenticatedContext().firestore(),env.authenticatedContext('stranger').firestore(),env.authenticatedContext('other').firestore()]) {
    await assertFails(getDoc(doc(db,'users/owner/portfolios/sujin')));
    await assertFails(setDoc(doc(db,'users/owner/portfolios/sujin'),sujinProfile()));
  }
});
test('수진 수량·전략·금액·추가 필드를 검증한다',async()=>{
  const p=sujinProfile(),ref=doc(owner(),'users/owner/portfolios/sujin');
  for(const update of [{cash:NaN},{totalBalance:-1},{strategy:'haa'},{quantityMode:'short'},{isAdmin:true},{holdings:{...p.holdings,QQQM:-1}}]) await assertFails(setDoc(ref,{...p,...update}));
  await assertFails(deleteDoc(ref));
});
test('수진 계산안은 가격 시각과 정상 수량으로 추가만 가능하다',async()=>{
  const ref=doc(owner(),'users/owner/portfolios/sujin/rebalances/one'),data=sujinRecord();
  await assertSucceeds(setDoc(ref,data));
  await assertFails(updateDoc(ref,{equity:1}));await assertFails(deleteDoc(ref));
  await assertFails(getDoc(doc(env.authenticatedContext('other').firestore(),'users/owner/portfolios/sujin/rebalances/one')));
  await assertFails(setDoc(doc(owner(),'users/owner/portfolios/sujin/rebalances/bad'),{...data,lines:data.lines.map(l=>({...l,target:-1}))}));
  await assertFails(setDoc(doc(owner(),'users/owner/portfolios/sujin/rebalances/missing'),{...data,lines:[]}));
  for(const update of [{held:NaN},{price:Infinity},{price:0},{target:'10'},{trade:false},{trade:Infinity},{held:-1},{target:1e10},{ticker:'FAKE'},{extra:1}]) {
    await assertFails(setDoc(doc(owner(),'users/owner/portfolios/sujin/rebalances/malformed'),{...data,lines:data.lines.map((l,i)=>i===8?{...l,...update}:l)}));
  }
});
test('수진 가격·잠금·장기 캐시를 클라이언트가 조작하지 못한다',async()=>{
  for(const path of ['market/sujin','status/sujinQuotes','internal/sujinHistory','internal/sujinQuotesLease','internal/sujinHistoryLease']) await assertFails(setDoc(doc(owner(),path),{payload:'fake'}));
  await assertFails(getDoc(doc(owner(),'internal/sujinHistory')));
});

test('공유 구성원은 두 포트폴리오와 이력을 같은 소유자 경로에서 사용한다',async()=>{
  const wife=env.authenticatedContext('wife').firestore();
  await assertSucceeds(setDoc(doc(wife,'users/owner'),profile()));
  await assertSucceeds(getDoc(doc(wife,'users/owner')));
  await assertSucceeds(setDoc(doc(wife,'users/owner/portfolios/sujin'),sujinProfile()));
  await assertSucceeds(setDoc(doc(wife,'users/owner/portfolios/sujin/rebalances/shared'),sujinRecord()));
  await assertSucceeds(getDoc(doc(owner(),'users/owner/portfolios/sujin/rebalances/shared')));
  await assertSucceeds(setDoc(doc(wife,'users/owner/rebalances/shared'),{createdAt:serverTimestamp(),signalMonth:'2026-07',sp500:'SPY',equity:10000,kind:'계산안',lines:[]}));
  await assertFails(deleteDoc(doc(wife,'users/owner')));
  await assertFails(getDoc(doc(wife,'users/other')));
  await assertFails(setDoc(doc(wife,'users/wife'),profile()));
});

test('클라이언트 관리자는 물론 구성원도 권한·초대·승인·감사 기록을 직접 변경할 수 없다',async()=>{
  for(const uid of ['owner','wife','stranger']) {
    const db=env.authenticatedContext(uid).firestore();
    for(const path of ['access/wife','internal/family','familyInvites/invite','familyRequests/wife','familyAudit/one']) {
      await assertFails(setDoc(doc(db,path),{enabled:true,role:'admin',ownerUid:uid}));
      if(path!=='access/'+uid)await assertFails(getDoc(doc(db,path)));
    }
  }
});

test('이미 사용 중인 구성원도 접근 해제 직후 서버 읽기·쓰기가 거부된다',async()=>{
  const wife=env.authenticatedContext('wife').firestore();
  await assertSucceeds(setDoc(doc(wife,'users/owner'),profile()));
  await env.withSecurityRulesDisabled(ctx=>updateDoc(doc(ctx.firestore(),'access/wife'),{enabled:false}));
  await assertFails(getDoc(doc(wife,'users/owner')));
  await assertFails(setDoc(doc(wife,'users/owner'),profile()));
  await assertFails(getDoc(doc(wife,'market/sujin')));
  await assertFails(setDoc(doc(wife,'users/owner/portfolios/sujin'),sujinProfile()));
});

test('HAA 구문서와 정수·소수점 설정을 모두 저장하고 잘못된 단위는 생성·갱신에서 거부한다',async()=>{
  const ref=doc(owner(),'users/owner');
  await assertSucceeds(setDoc(ref,profile()));
  for(const quantityMode of ['whole','fractional']){
    await assertSucceeds(setDoc(ref,{...profile(),quantityMode}));
    await assertSucceeds(updateDoc(ref,{quantityMode,updatedAt:serverTimestamp()}));
  }
  for(const quantityMode of [null,0,true,'unsupported']){
    await assertSucceeds(deleteDoc(ref));
    await assertFails(setDoc(doc(owner(),'users/owner'),{...profile(),quantityMode}));
    await assertSucceeds(setDoc(ref,profile()));
    await assertFails(updateDoc(ref,{quantityMode,updatedAt:serverTimestamp()}));
  }
  await assertSucceeds(updateDoc(ref,{quantityMode:deleteField(),updatedAt:serverTimestamp()}));
});

test('HAA 계산안은 구형·신형 수량 단위를 보존하고 이력 변경은 허용하지 않는다',async()=>{
  const data={createdAt:serverTimestamp(),signalMonth:'2026-10',sp500:'SPYM',equity:10000,kind:'계산안',lines:[]};
  for(const [id,quantityMode] of [['legacy',undefined],['whole','whole'],['fractional','fractional']]){
    const ref=doc(owner(),'users/owner/rebalances/'+id);
    await assertSucceeds(setDoc(ref,{...data,...(quantityMode?{quantityMode}:{})}));
    await assertSucceeds(getDoc(ref));
    await assertFails(updateDoc(ref,{quantityMode:'fractional'}));
  }
  for(const quantityMode of [null,true,'0.01'])await assertFails(setDoc(doc(owner(),'users/owner/rebalances/bad-mode'),{...data,quantityMode}));
});

test('공유 구성원은 연금 설정과 계산안을 같은 소유자 경로에 저장하고 최근 이력을 조회한다',async()=>{
  const wife=env.authenticatedContext('wife').firestore(),ref=doc(wife,'users/owner/portfolios/pension');
  await assertSucceeds(setDoc(doc(owner(),'users/owner'),profile()));
  await assertSucceeds(setDoc(doc(owner(),'users/owner/portfolios/sujin'),sujinProfile()));
  await assertSucceeds(setDoc(ref,pensionProfile()));
  await assertSucceeds(updateDoc(ref,{'targetOverrides.379800':0,updatedAt:serverTimestamp()}));
  await assertSucceeds(updateDoc(ref,{'targetOverrides.379800':null,updatedAt:serverTimestamp()}));
  await assertSucceeds(getDoc(doc(owner(),'users/owner/portfolios/pension')));
  await assertSucceeds(setDoc(doc(wife,'users/owner/portfolios/pension/rebalances/shared'),pensionRecord()));
  await assertSucceeds(getDocs(query(collection(owner(),'users/owner/portfolios/pension/rebalances'),orderBy('createdAt','desc'),limit(24))));
  await assertFails(deleteDoc(ref));
  const haa=await getDoc(doc(owner(),'users/owner')),sujin=await getDoc(doc(owner(),'users/owner/portfolios/sujin'));
  if(haa.data().cash!==10000||sujin.data().strategy!=='sujin-fixed-v1')throw new Error('연금 저장이 다른 포트폴리오를 변경했습니다.');
});

test('미인증·미승인·다른 가족은 연금 설정과 이력을 읽거나 변경할 수 없다',async()=>{
  await assertSucceeds(setDoc(doc(owner(),'users/owner/portfolios/pension'),pensionProfile()));
  await assertSucceeds(setDoc(doc(owner(),'users/owner/portfolios/pension/rebalances/one'),pensionRecord()));
  for(const db of [env.unauthenticatedContext().firestore(),env.authenticatedContext('stranger').firestore(),env.authenticatedContext('other').firestore()]){
    await assertFails(getDoc(doc(db,'users/owner/portfolios/pension')));
    await assertFails(getDocs(collection(db,'users/owner/portfolios/pension/rebalances')));
    await assertFails(getDoc(doc(db,'users/owner/portfolios/pension/rebalances/one')));
    await assertFails(setDoc(doc(db,'users/owner/portfolios/pension'),pensionProfile()));
    await assertFails(updateDoc(doc(db,'users/owner/portfolios/pension'),{principal:1,updatedAt:serverTimestamp()}));
    await assertFails(setDoc(doc(db,'users/owner/portfolios/pension/rebalances/new'),pensionRecord()));
    await assertFails(deleteDoc(doc(db,'users/owner/portfolios/pension')));
  }
  const wife=env.authenticatedContext('wife').firestore();
  await assertFails(setDoc(doc(wife,'users/wife/portfolios/pension'),pensionProfile()));
  await assertFails(setDoc(doc(wife,'users/other/portfolios/pension'),pensionProfile()));
});

test('연금 설정은 생성·덮어쓰기·부분 갱신에서 타입·필수 필드·범위를 모두 검증한다',async()=>{
  const p=pensionProfile(),ref=doc(owner(),'users/owner/portfolios/pension');
  await assertSucceeds(setDoc(ref,p));
  const missingHolding={...p.holdings},missingWeight={...p.weights},missingOverride={...p.targetOverrides};
  delete missingHolding['449170'];delete missingWeight['449170'];delete missingOverride['449170'];
  const invalid=[{currency:'USD'},{strategy:'haa'},{totalBalance:-1},{totalBalance:Infinity},{principal:NaN},{principal:1e13},
    {role:'admin'},{ownerUid:'other'},{extra:'x'.repeat(10000)},{updatedAt:'2026-10-26T06:30:00Z'},
    {holdings:{...p.holdings,'379800':.5}},{holdings:{...p.holdings,'379800':-1}},{holdings:{...p.holdings,'379800':1e10}},
    {holdings:{...p.holdings,FAKE:1}},{weights:{...p.weights,'379800':.23}},{weights:{...p.weights,'379800':NaN}},
    {weights:{...p.weights,'379800':Infinity}},{weights:{...p.weights,'379800':true}},{weights:{...p.weights,'379800':'0.24'}},
    {targetOverrides:{...p.targetOverrides,'379800':.5}},{targetOverrides:{...p.targetOverrides,'379800':false}},
    {targetOverrides:{...p.targetOverrides,'379800':-1}},{targetOverrides:{...p.targetOverrides,'379800':'0'}},
    {holdings:missingHolding},{weights:missingWeight},{targetOverrides:missingOverride}];
  for(const patch of invalid){
    await env.withSecurityRulesDisabled(ctx=>deleteDoc(doc(ctx.firestore(),'users/owner/portfolios/pension')));
    await assertFails(setDoc(ref,{...p,...patch}));
    await assertSucceeds(setDoc(ref,p));
    await assertFails(setDoc(ref,{...p,...patch}));
    await assertFails(updateDoc(ref,{...patch,...('updatedAt' in patch?{}:{updatedAt:serverTimestamp()})}));
  }
  for(const key of ['strategy','currency','totalBalance','principal','holdings','weights','targetOverrides','updatedAt']){
    const missing={...p};delete missing[key];
    await assertFails(setDoc(ref,missing));
    await assertFails(updateDoc(ref,{[key]:deleteField(),...('updatedAt'===key?{}:{updatedAt:serverTimestamp()})}));
  }
});

test('연금 계산안은 고정 9개 행의 정수·시세를 검증하며 추가 후 변경·삭제를 차단한다',async()=>{
  const data=pensionRecord(),ref=doc(owner(),'users/owner/portfolios/pension/rebalances/one');
  await assertSucceeds(setDoc(ref,data));
  await assertFails(updateDoc(ref,{equity:2000000}));
  await assertFails(deleteDoc(ref));
  const malformed=[{held:NaN},{held:.5},{held:-1},{price:0},{price:NaN},{price:Infinity},{price:'10000'},{price:true},
    {target:'5'},{target:5.5},{trade:false},{trade:0},{trade:Infinity},{ticker:'FAKE'},{extra:1}];
  for(const [i,patch] of malformed.entries())await assertFails(setDoc(doc(owner(),'users/owner/portfolios/pension/rebalances/malformed-'+i),{...data,lines:data.lines.map((line,index)=>index===8?{...line,...patch}:line)}));
  // 9개 위치 모두 검증하는지 확인한다. 종목·매매수량 필드는 저장하지 않는다.
  for(let index=0;index<9;index++)await assertFails(setDoc(doc(owner(),'users/owner/portfolios/pension/rebalances/row-'+index),{...data,lines:data.lines.map((line,i)=>i===index?{...line,target:.5}:line)}));
  const missingPrice={...data.lines[0]};delete missingPrice.price;
  for(const [i,patch] of [{lines:data.lines.slice(0,8)},{lines:[...data.lines,data.lines[0]]},{lines:[missingPrice,...data.lines.slice(1)]},{createdAt:'2026-10-26T06:30:00Z'},
    {currency:'USD'},{remainingCash:1000001},{priceUpdatedAt:'invalid'},{oldestPriceAt:'x'.repeat(10000)},
    {weights:{...data.weights,'379800':.25}},{role:'admin'}].entries())await assertFails(setDoc(doc(owner(),'users/owner/portfolios/pension/rebalances/schema-'+i),{...data,...patch}));
  const missing={...data};delete missing.principal;
  await assertFails(setDoc(doc(owner(),'users/owner/portfolios/pension/rebalances/missing'),missing));
});

test('연금 접근도 구성원 해제 직후 거부하며 가격·상태·잠금 문서는 직접 쓸 수 없다',async()=>{
  const wife=env.authenticatedContext('wife').firestore();
  await assertSucceeds(setDoc(doc(wife,'users/owner/portfolios/pension'),pensionProfile()));
  for(const path of ['market/pension','status/pensionQuotes','internal/pensionQuotesLease'])await assertFails(setDoc(doc(wife,path),{prices:{'379800':1}}));
  await assertFails(getDoc(doc(wife,'internal/pensionQuotesLease')));
  await env.withSecurityRulesDisabled(ctx=>updateDoc(doc(ctx.firestore(),'access/wife'),{enabled:false}));
  await assertFails(getDoc(doc(wife,'users/owner/portfolios/pension')));
  await assertFails(setDoc(doc(wife,'users/owner/portfolios/pension'),pensionProfile()));
  await assertFails(setDoc(doc(wife,'users/owner/portfolios/pension/rebalances/after-revoke'),pensionRecord()));
  await assertFails(getDoc(doc(wife,'market/pension')));
});
