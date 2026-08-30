import { initializeApp } from 'firebase/app';
import { connectAuthEmulator, getAuth, GoogleAuthProvider, signInWithPopup, signOut } from 'firebase/auth';
import { connectFirestoreEmulator, getFirestore } from 'firebase/firestore';
import { getToken, initializeAppCheck, ReCaptchaV3Provider, type AppCheck } from 'firebase/app-check';

const env = import.meta.env;
export const demo = env.VITE_DEMO_MODE === 'true';
export const configured = ['API_KEY','AUTH_DOMAIN','PROJECT_ID','APP_ID'].every(k => {
  const value = env[`VITE_FIREBASE_${k}`];
  return value && !value.startsWith('YOUR_');
});
const app = configured && !demo ? initializeApp({
  apiKey: env.VITE_FIREBASE_API_KEY, authDomain: env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: env.VITE_FIREBASE_PROJECT_ID, appId: env.VITE_FIREBASE_APP_ID,
}) : null;
export const auth = app ? getAuth(app) : null;
export const db = app ? getFirestore(app) : null;
let appCheck: AppCheck | null = null;
if (app && env.VITE_RECAPTCHA_SITE_KEY) {
  appCheck = initializeAppCheck(app, { provider: new ReCaptchaV3Provider(env.VITE_RECAPTCHA_SITE_KEY), isTokenAutoRefreshEnabled: true });
}
if (auth && db && env.VITE_USE_EMULATORS === 'true') {
  if (!env.VITE_FIREBASE_PROJECT_ID.startsWith('demo-')) throw new Error('에뮬레이터에는 demo- 프로젝트만 사용하세요.');
  connectAuthEmulator(auth, 'http://127.0.0.1:9099');
  connectFirestoreEmulator(db, '127.0.0.1', 8080);
}
export async function login() { if (auth) await signInWithPopup(auth, new GoogleAuthProvider()); }
export async function logout() { if (auth) await signOut(auth); }
export async function apiHeaders() {
  if (!auth?.currentUser) throw new Error('로그인이 필요합니다.');
  const headers: Record<string, string> = { Authorization: `Bearer ${await auth.currentUser.getIdToken()}` };
  if (appCheck) headers['X-Firebase-AppCheck'] = (await getToken(appCheck)).token;
  return headers;
}
