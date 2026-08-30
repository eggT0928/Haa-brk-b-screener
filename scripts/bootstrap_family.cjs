// 최초 가족 관리자만 신뢰된 CLI 인증으로 설정한다. 초대·일반 승인은 이후 웹앱에서 처리한다.
const {getGlobalDefaultAccount}=require('firebase-tools/lib/auth');
const {requireAuth}=require('firebase-tools/lib/requireAuth');
const {findUser}=require('firebase-tools/lib/gcp/auth');
const {Client}=require('firebase-tools/lib/apiv2');

async function main(){
  const args=process.argv.slice(2);
  const value=name=>args[args.indexOf(name)+1];
  if(!args.includes('--project')||!args.includes('--owner-email'))throw new Error('--project 및 --owner-email이 필요합니다. 기본은 읽기 점검이며 --apply 지정 시만 적용합니다.');
  const project=value('--project'),email=value('--owner-email').trim().toLowerCase();
  if(!/^[a-z][a-z0-9-]{4,28}[a-z0-9]$/.test(project)||!email.includes('@'))throw new Error('프로젝트·이메일을 확인하세요.');
  await requireAuth({...getGlobalDefaultAccount(),project,nonInteractive:true});
  const user=await findUser(project,email);
  if(user.email?.toLowerCase()!==email||user.emailVerified!==true||user.disabled||!user.providerUserInfo?.some(p=>p.providerId==='google.com'))throw new Error('먼저 관리자가 검증된 Google 계정으로 로그인해야 합니다.');
  const db=new Client({urlPrefix:'https://firestore.googleapis.com',apiVersion:'v1'});
  const base=`/projects/${project}/databases/(default)/documents`;
  async function get(path){try{return(await db.get(base+'/'+path)).body;}catch(e){if(e.status===404||e.context?.response?.statusCode===404)return null;throw e;}}
  const config=await get('internal/family'),grant=await get('access/'+user.uid);
  if(config&&(config.fields?.ownerUid?.stringValue!==user.uid||config.fields?.ownerEmail?.stringValue!==email))throw new Error('다른 가족 관리자가 설정되어 있습니다. 변경하지 않습니다.');
  if(grant?.fields?.enabled?.booleanValue!==true)throw new Error('기존 승인 계정만 초기 관리자로 설정합니다. 최초 환경은 콘솔에서 access/{UID}.enabled=true 승인 후 다시 실행하세요.');
  if(grant.fields.ownerUid&&grant.fields.ownerUid.stringValue!==user.uid)throw new Error('이미 다른 공유 공간에 연결된 계정입니다. 변경하지 않습니다.');
  const ready=!!config&&grant.fields.role?.stringValue==='admin'&&grant.fields.ownerUid?.stringValue===user.uid&&grant.fields.email?.stringValue===email;
  if(!args.includes('--apply')||ready){console.log(JSON.stringify({project,email,uid:user.uid,ready,mode:'read-only',portfoliosChanged:false}));return;}
  const fields={enabled:{booleanValue:true},role:{stringValue:'admin'},ownerUid:{stringValue:user.uid},email:{stringValue:email}};
  const configFields={ownerUid:{stringValue:user.uid},ownerEmail:{stringValue:email}};
  // 읽은 문서의 갱신 시각을 선행 조건으로 삼아 다른 관리자의 변경을 덮어쓰지 않는다.
  await db.post(`/projects/${project}/databases/(default)/documents:commit`,{writes:[
    {update:{name:base.slice(1)+'/internal/family',fields:configFields},updateMask:{fieldPaths:Object.keys(configFields)},currentDocument:config?{updateTime:config.updateTime}:{exists:false}},
    {update:{name:base.slice(1)+'/access/'+user.uid,fields},updateMask:{fieldPaths:Object.keys(fields)},currentDocument:{updateTime:grant.updateTime}}
  ]});
  const checked=await get('access/'+user.uid);
  if(checked.fields?.role?.stringValue!=='admin'||checked.fields?.ownerUid?.stringValue!==user.uid)throw new Error('관리자 설정 재확인에 실패했습니다.');
  console.log(JSON.stringify({project,email,uid:user.uid,ready:true,portfoliosChanged:false}));
}
main().catch(e=>{console.error(e.message);process.exitCode=1;});
