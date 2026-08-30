import { readFile } from 'node:fs/promises';
import { after, before, beforeEach, test } from 'node:test';
import { initializeTestEnvironment, assertFails, assertSucceeds } from '@firebase/rules-unit-testing';
import { collection, deleteDoc, doc, getDoc, getDocs, serverTimestamp, setDoc, updateDoc } from 'firebase/firestore';

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
    await setDoc(doc(db,'signals/confirmed'),{month:'2026-07'});
    await setDoc(doc(db,'internal/history'),{payload:'비공개 캐시'});
  });
});
after(async () => { await env?.cleanup(); });
const owner = () => env.authenticatedContext('owner').firestore();
const sujinTickers=['QQQM','SPY','SPYM','JEPQ','BRK-B','IEF','TLT','GLD','PDBC'];
const sujinProfile=()=>({strategy:'sujin-fixed-v1',holdings:Object.fromEntries(sujinTickers.map(t=>[t,0])),totalMode:'manual',totalBalance:10000,cash:0,quantityMode:'fractional',backtestStart:'',backtestEnd:'',updatedAt:serverTimestamp()});
const sujinRecord=()=>({createdAt:serverTimestamp(),month:'2026-08',strategy:'sujin-fixed-v1',kind:'계산안',equity:10000,remainingCash:0,priceUpdatedAt:'2026-08-28T21:30:00Z',oldestPriceAt:'2026-08-28T20:00:00Z',quantityMode:'fractional',lines:sujinTickers.map(ticker=>({ticker,held:0,price:100,target:10,trade:10}))});

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
