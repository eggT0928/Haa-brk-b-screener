import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const fixture=vi.hoisted(()=>({request:vi.fn()}));
vi.mock('./familyAccess',()=>({accessRequest:fixture.request}));
import { FamilyAccessPanel } from './FamilyAccessPanel';
const listing={members:[{id:'owner',email:'owner@example.com',role:'admin',enabled:true},{id:'wife',email:'wife@example.com',role:'member',enabled:true}],invites:[],requests:[{id:'new',email:'new@example.com',status:'pending'}]};
beforeEach(()=>{fixture.request.mockReset();fixture.request.mockImplementation(async(_path,body)=>body?{message:'처리 완료'}:listing);});
afterEach(cleanup);

it('이메일 초대는 공유 편집 동의를 받은 후에만 등록한다',async()=>{
  render(<FamilyAccessPanel/>);await screen.findByText('owner@example.com');
  fireEvent.change(screen.getByLabelText('초대할 Google 이메일'),{target:{value:'near9600@gmail.com'}});
  const button=screen.getByRole('button',{name:'이메일 초대 등록'}) as HTMLButtonElement;
  expect(button.disabled).toBe(true);fireEvent.click(screen.getByRole('checkbox'));fireEvent.click(button);
  await waitFor(()=>expect(fixture.request).toHaveBeenCalledWith('manage',{action:'invite',email:'near9600@gmail.com'}));
  expect(screen.getByText(/이메일은 자동 발송되지 않습니다/)).toBeTruthy();
});
it('접근 해제는 확인을 거치고 관리자 자기 해제 버튼은 없다',async()=>{
  render(<FamilyAccessPanel/>);await screen.findByText('wife@example.com');
  expect(screen.getAllByRole('button',{name:'접근 해제'}).length).toBe(1);
  fireEvent.click(screen.getByRole('button',{name:'접근 해제'}));
  expect(fixture.request).not.toHaveBeenCalledWith('manage',expect.objectContaining({action:'revoke'}));
  fireEvent.click(screen.getByRole('button',{name:'변경 확인'}));
  await waitFor(()=>expect(fixture.request).toHaveBeenCalledWith('manage',{action:'revoke',uid:'wife'}));
});
it('승인과 거절 확인을 취소하면 서버 권한을 변경하지 않는다',async()=>{
  render(<FamilyAccessPanel/>);await screen.findByText('new@example.com');
  fireEvent.click(screen.getByRole('button',{name:'승인'}));
  fireEvent.click(screen.getByRole('button',{name:'취소'}));
  expect(fixture.request).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole('button',{name:'거절'}));
  fireEvent.click(screen.getByRole('button',{name:'변경 확인'}));
  await waitFor(()=>expect(fixture.request).toHaveBeenCalledWith('manage',{action:'reject',uid:'new'}));
});
it('관리 권한이 없으면 서버 거부를 안내한다',async()=>{
  fixture.request.mockRejectedValue(new Error('관리자만 가능합니다.'));
  render(<FamilyAccessPanel/>);expect((await screen.findByRole('alert')).textContent).toContain('관리자만 가능합니다.');
});
