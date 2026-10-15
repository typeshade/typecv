import assert from 'node:assert/strict';
import test from 'node:test';
import { reconstructTwoView, triangulate } from '../src/index.ts';
import { multiply, symmetricEigen, transpose } from '../src/sfm/linear.ts';

const K = {fx:1000,fy:990,cx:640,cy:360};
function rY(a: number): Float64Array { return Float64Array.from([Math.cos(a),0,Math.sin(a),0,1,0,-Math.sin(a),0,Math.cos(a)]); }
function posePoint(R: Float64Array,t: number[],X: number[]): number[]{
  return [R[0]!*X[0]!+R[1]!*X[1]!+R[2]!*X[2]!+t[0]!,R[3]!*X[0]!+R[4]!*X[1]!+R[5]!*X[2]!+t[1]!,R[6]!*X[0]!+R[7]!*X[1]!+R[8]!*X[2]!+t[2]!];
}
function pixel(X:number[]):[number,number] { return [K.fx * X[0]!/X[2]! + K.cx, K.fy * X[1]!/X[2]! + K.cy]; }
let state = 1337;
function rnd(){state ^= state<<13; state ^= state>>>17; state ^= state<<5;return (state>>>0)/2**32;}
function synthetic(n:number,noisePx:number,outlierRatio:number){
  state=1337;
  const R=rY(0.09),t=[0.42,-0.035,0.065];
  const matches=new Float64Array(n*4);
  for(let i=0;i<n;i++){
    const X=[(rnd()-.5)*4,(rnd()-.5)*2.5,3+rnd()*7];
    const p=pixel(X),q=pixel(posePoint(R,t,X));
    matches.set([
      p[0]+(rnd()-.5)*noisePx*2,p[1]+(rnd()-.5)*noisePx*2,
      q[0]+(rnd()-.5)*noisePx*2,q[1]+(rnd()-.5)*noisePx*2,
    ],i*4);
    if(rnd()<outlierRatio){matches[i*4+2]=rnd()*1280;matches[i*4+3]=rnd()*720;}
  }
  return {R,t,matches};
}

// Matrix primitives prove we have no hidden external numerics dependency.
test('symmetric Jacobi eigensystem reconstructs input', () => {
  const A=Float64Array.from([5,1,2, 1,4,0.5, 2,0.5,3]);
  const {values,vectors}=symmetricEigen(A,3);
  const reconstructed=multiply(multiply(vectors,Float64Array.from([values[0]!,0,0,0,values[1]!,0,0,0,values[2]!]),3),transpose(vectors,3),3);
  for(let i=0;i<9;i++)assert.ok(Math.abs(reconstructed[i]! - A[i]!)<1e-10);
});
test('triangulation reconstructs a known 3D point',()=>{
  const R=rY(0.07),t=[0.4,0.01,0.03],X=[0.5,-0.4,5];
  const X2=posePoint(R,t,X);
  const p=triangulate(X[0]/X[2],X[1]/X[2],X2[0]/X2[2],X2[1]/X2[2],R,t);
  assert.ok(p);
  for(let i=0;i<3;i++)assert.ok(Math.abs(p[i]!-X[i]!)<1e-7,`${i} => ${p[i]} vs ${X[i]}`);
});
test('two-view pose and sparse points from clean synthetic pixel matches',()=>{
  const {R,t,matches}=synthetic(70,0,0);
  const result=reconstructTwoView(matches,K,K,{iterations:90,threshold:0.001,seed:9});
  const expectedDir=t.map(x=>x/Math.hypot(...t));
  const cos=expectedDir.reduce((sum,x,i)=>sum+x*result.t[i]!,0);
  const Rot= multiply(result.R,transpose(R,3),3);
  const rotAngle=Math.acos(Math.max(-1,Math.min(1,(Rot[0]!+Rot[4]!+Rot[8]!-1)/2)));
  assert.ok(result.inlierCount >= 67,`inliers=${result.inlierCount}`);
  assert.ok(result.positiveDepthCount >= 67,`positive=${result.positiveDepthCount}`);
  assert.ok(cos > 0.999,`direction=${cos}`);
  assert.ok(rotAngle < 1e-3,`rotation error=${rotAngle}`);
  assert.ok(result.meanReprojectionErrorPx < 0.01,`err=${result.meanReprojectionErrorPx}`);
});
test('RANSAC tolerates outliers and noise',()=>{
  const {R,t,matches}=synthetic(160,0.28,0.18);
  const result=reconstructTwoView(matches,K,K,{iterations:650,threshold:0.0018,seed:71});
  const expectedDir=t.map(x=>x/Math.hypot(...t));
  const cos=expectedDir.reduce((sum,x,i)=>sum+x*result.t[i]!,0);
  const Rot=multiply(result.R,transpose(R,3),3);
  const rotAngle=Math.acos(Math.max(-1,Math.min(1,(Rot[0]!+Rot[4]!+Rot[8]!-1)/2)));
  console.log(JSON.stringify({inliers:result.inlierCount,valid:result.positiveDepthCount,reprojectionPx:result.meanReprojectionErrorPx,rotationErrorDeg:rotAngle*180/Math.PI,translationCosine:cos}));
  assert.ok(result.inlierCount >= 105,`inliers=${result.inlierCount}`);
  assert.ok(result.positiveDepthCount >= 100,`positive=${result.positiveDepthCount}`);
  assert.ok(cos > 0.85,`direction=${cos}`);
  assert.ok(rotAngle < 0.05,`rotation error=${rotAngle}`);
  assert.ok(result.meanReprojectionErrorPx < 2,`err=${result.meanReprojectionErrorPx}`);
});
test('rejects invalid inputs',()=>{
  assert.throws(()=>reconstructTwoView(new Float64Array(12),K,K),/8 finite/);
  assert.throws(()=>reconstructTwoView(new Float64Array(32),{...K,fx:0},K),/intrinsics/);
});
test('camera 2 may have different intrinsics',()=>{
  const localK2={fx:770,fy:775,cx:318,cy:245};
  state=8453;
  const R=rY(-0.075),t=[0.38,0.06,0.02];
  const matches=new Float64Array(96*4);
  for(let i=0;i<96;i++){
    const X=[(rnd()-.5)*3,(rnd()-.5)*2,4+rnd()*6];
    const A=pixel(X), X2=posePoint(R,t,X);
    const B=[localK2.fx*X2[0]!/X2[2]!+localK2.cx,localK2.fy*X2[1]!/X2[2]!+localK2.cy];
    matches.set([A[0],A[1],B[0],B[1]],i*4);
  }
  const result=reconstructTwoView(matches,K,localK2,{iterations:80,threshold:0.0005});
  assert.ok(result.inlierCount>=92);
  assert.ok(result.meanReprojectionErrorPx<0.01);
});
test('rejects coincident 2D feature positions',()=>{
  const same=new Float64Array(50*4);
  for(let i=0;i<50;i++)same.set([100,200,110,210],i*4);
  assert.throws(()=>reconstructTwoView(same,K,K,{iterations:12}),/valid two-view geometry/);
});

test('adaptive stopping is deterministic and honors its bounds',()=>{
  const {matches}=synthetic(100,0,0);
  const opts={iterations:500,minIterations:24,seed:18};
  const a=reconstructTwoView(matches,K,K,opts), b=reconstructTwoView(matches,K,K,opts);
  assert.ok(a.diagnostics.ransacIterations>=24 && a.diagnostics.ransacIterations<500);
  assert.deepEqual(a,b);
  const capped=reconstructTwoView(matches,K,K,{iterations:5,minIterations:5});
  assert.equal(capped.diagnostics.ransacIterations,5);
});

test('nonlinear refinement reduces error on the same observations and preserves pose constraints',()=>{
  const {matches}=synthetic(160,0.28,0.18);
  const options={iterations:650,threshold:0.0018,seed:71};
  const initial=reconstructTwoView(matches,K,K,{...options,refinementIterations:0});
  const refined=reconstructTwoView(matches,K,K,options);
  assert.equal(initial.diagnostics.refinementApplied,false);
  assert.ok(refined.diagnostics.refinementApplied);
  assert.ok(refined.diagnostics.meanReprojectionErrorAfterPx < refined.diagnostics.meanReprojectionErrorBeforePx*0.8);
  assert.ok(refined.meanReprojectionErrorPx < initial.meanReprojectionErrorPx);
  assert.ok(Math.abs(Math.hypot(...refined.t)-1)<1e-10);
  const identity=multiply(refined.R,transpose(refined.R,3),3);
  for(let i=0;i<9;i++)assert.ok(Math.abs(identity[i]!-(i%4===0?1:0))<1e-9);
  const [x,y,z]=refined.t;
  const E=multiply(Float64Array.of(0,-z!,y!,z!,0,-x!,-y!,x!,0),refined.R,3);
  const norm=Math.hypot(...E);
  for(let i=0;i<9;i++)assert.ok(Math.abs(E[i]!/norm-refined.E[i]!)<1e-10);
  console.log(JSON.stringify({beforePx:initial.meanReprojectionErrorPx,afterPx:refined.meanReprojectionErrorPx,diagnostics:refined.diagnostics}));
});

function scene(baseline:number,planar=false){
  state=937;
  const R=rY(0.08),t=[baseline,0,0],matches=new Float64Array(100*4);
  for(let i=0;i<100;i++){
    const X=[(rnd()-.5)*3,(rnd()-.5)*2,planar?6:4+rnd()*6];
    matches.set([...pixel(X),...pixel(posePoint(R,t,X))],4*i);
  }
  return matches;
}

test('rejects tiny baselines and pure rotation rather than returning unstable depth',()=>{
  assert.throws(()=>reconstructTwoView(scene(0.0001),K,K,{iterations:100}),/Low parallax|Cheirality|valid two-view/);
  assert.throws(()=>reconstructTwoView(scene(0),K,K,{iterations:100}),/Low parallax|Cheirality|valid two-view/);
});

test('rejects exact planar and collinear eight-point degeneracies',()=>{
  assert.throws(()=>reconstructTwoView(scene(0.4,true),K,K,{iterations:100}),/valid two-view/);
  const matches=new Float64Array(40*4);
  for(let i=0;i<40;i++)matches.set([i*10,100,i*11+20,120],i*4);
  assert.throws(()=>reconstructTwoView(matches,K,K,{iterations:40}),/valid two-view/);
});

test('quality mask excludes inaccurate points and leaves NaN triplets',()=>{
  const {matches}=synthetic(160,0.5,0.18);
  const result=reconstructTwoView(matches,K,K,{iterations:650,threshold:0.002,seed:71,maxReprojectionErrorPx:0.2});
  assert.ok(result.positiveDepthCount < result.inlierCount);
  assert.equal(result.validPoints.reduce((a,b)=>a+b,0),result.positiveDepthCount);
  for(let i=0;i<result.validPoints.length;i++){
    const X=result.points.slice(i*3,i*3+3);
    if(!result.validPoints[i]){assert.ok(X.every(Number.isNaN));continue;}
    assert.equal(result.inliers[i],1);
    const first=pixel(Array.from(X)),second=pixel(posePoint(result.R,Array.from(result.t),Array.from(X)));
    assert.ok(Math.hypot(first[0]-matches[i*4]!,first[1]-matches[i*4+1]!)<=0.2+1e-9);
    assert.ok(Math.hypot(second[0]-matches[i*4+2]!,second[1]-matches[i*4+3]!)<=0.2+1e-9);
  }
});

test('validates new stability controls',()=>{
  const {matches}=synthetic(20,0,0);
  for(const options of [{confidence:1},{confidence:NaN},{minIterations:0},{iterations:10,minIterations:11},
    {refinementIterations:-1},{huberPx:0},{minParallaxDeg:-1},{maxReprojectionErrorPx:Infinity},{seed:NaN}])
    assert.throws(()=>reconstructTwoView(matches,K,K,options),RangeError);
});

test('refinement remains stable across noisy scenes and RANSAC seeds',()=>{
  for(const seed of [3,17,91]){
    const {R,t,matches}=synthetic(180,0.7,0.3);
    const result=reconstructTwoView(matches,K,K,{iterations:1200,threshold:0.002,seed});
    const delta=multiply(result.R,transpose(R,3),3);
    const angle=Math.acos(Math.max(-1,Math.min(1,(delta[0]!+delta[4]!+delta[8]!-1)/2)));
    const direction=t.reduce((sum,x,i)=>sum+x*result.t[i]!,0)/Math.hypot(...t);
    assert.ok(angle<0.02,`seed=${seed}, angle=${angle}`);
    assert.ok(direction>0.97,`seed=${seed}, direction=${direction}`);
    assert.ok(result.diagnostics.meanReprojectionErrorAfterPx<=result.diagnostics.meanReprojectionErrorBeforePx);
    assert.ok(result.positiveDepthCount>100);
  }
});

test('pixel-weighted refinement handles unequal anisotropic intrinsics',()=>{
  const other={fx:450,fy:1300,cx:320,cy:240};
  const {matches}=synthetic(120,0.4,0);
  for(let i=0;i<matches.length;i+=4){
    matches[i+2]=(matches[i+2]!-K.cx)/K.fx*other.fx+other.cx;
    matches[i+3]=(matches[i+3]!-K.cy)/K.fy*other.fy+other.cy;
  }
  const result=reconstructTwoView(matches,K,other,{iterations:300});
  assert.ok(result.diagnostics.refinementApplied);
  assert.ok(result.diagnostics.meanReprojectionErrorAfterPx<result.diagnostics.meanReprojectionErrorBeforePx);
  assert.ok(result.meanReprojectionErrorPx<0.5);
});
