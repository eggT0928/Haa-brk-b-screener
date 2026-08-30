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
