/* clozn · light — the shader rendering core (spec §9.2, "first light" productized).
   Raw WebGL2, hand-written GLSL, zero dependencies. One shader, two worlds (uNight = dual parity).
   MOTION = MEANING: uAct (0..1) gates the flow — idle is a near-still breath, activity makes the
   silk move. Callers raise activity on real events (generation, replay, interaction), never for mood.
   Fallback: no WebGL2 -> caller keeps its plain tokens.css ground (declared, quiet). */

const VS = `#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0,1);}`;

const FS = `#version 300 es
precision highp float;
uniform vec2 uRes; uniform float uTime; uniform vec2 uMouse; uniform float uNight; uniform float uAct;
out vec4 O;
float h21(vec2 p){p=fract(p*vec2(234.34,435.345));p+=dot(p,p+34.23);return fract(p.x*p.y);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(h21(i),h21(i+vec2(1,0)),f.x),mix(h21(i+vec2(0,1)),h21(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;
  for(int i=0;i<4;i++){v+=a*vnoise(p);p=p*2.03+vec2(1.7,9.2);a*=.55;}return v;}
const vec3 PINK=vec3(.910,.588,.839),GOLD=vec3(.941,.816,.588),
           PERI=vec3(.588,.686,.949),MINT=vec3(.431,.804,.745),LILAC=vec3(.729,.698,.886);
vec3 silk(vec2 uv,float sc,vec2 off,float fr,float spd,float ph,vec3 tint,vec2 mw,float t){
  vec2 q=uv*sc+off;
  vec2 w=vec2(fbm(q+t*spd),fbm(q+vec2(5.2,1.3)-t*spd*.8));
  vec2 p=q+2.1*w+mw*sc*.4;
  float band=sin(p.y*fr+4.0*w.x+ph);
  float th=pow(max(0.,1.-abs(band)),9.0);
  float bead=.6+.8*fbm(p*2.7+t*.2);
  float core=pow(max(0.,1.-abs(band)),24.0);
  return tint*th*bead+vec3(1)*core*.7;
}
void main(){
  vec2 uv=(gl_FragCoord.xy-.5*uRes)/uRes.y;
  float t=uTime;
  vec2 m=(uMouse-.5*uRes)/uRes.y;
  vec2 dm=uv-m; float md=exp(-dot(dm,dm)*6.0);
  vec2 mw=normalize(dm+1e-4)*md*.25*max(uAct,.25);
  /* silk intensity follows activity: a whisper at rest, a current when working */
  float amp=.18+.82*uAct;
  vec3 s=vec3(0.);
  s+=silk(uv,1.6,vec2(0.,8.),6.0,.05,0. ,PERI,mw,t)*.35;
  s+=silk(uv,1.2,vec2(3.,1.),5.0,.07,2.1,PINK,mw,t)*.45;
  s+=silk(uv,0.9,vec2(7.,4.),4.2,.06,4.4,MINT,mw,t)*.40;
  s+=silk(uv,0.7,vec2(2.,6.),3.4,.08,1.2,GOLD,mw,t)*.50;
  s*=amp;
  float lum=dot(s,vec3(.333));
  vec2 b1=vec2(.32*sin(t*.11),.10+.16*cos(t*.09));
  vec2 b2=vec2(-.38*cos(t*.07),-.14+.12*sin(t*.13));
  vec3 blooms=(vec3(1.,.98,.95)*(.0028/(dot(uv-b1,uv-b1)+.004))
              +LILAC*(.0022/(dot(uv-b2,uv-b2)+.004)))*(.35+.65*uAct);
  float vy=gl_FragCoord.y/uRes.y;
  vec3 bgD=mix(vec3(.902,.894,.953),vec3(.957,.965,.992),vy);
  vec3 bgN=mix(vec3(.047,.055,.135),vec3(.078,.075,.180),1.-vy);
  vec3 dawn=bgD*(1.-lum*.28)+s*.72+blooms*.5;
  vec3 night=bgN+s*.85+blooms;
  vec3 col=mix(dawn,night,uNight)*(1.-.22*dot(uv,uv));
  O=vec4(col,1.);
}`;

export function mountLight(canvas, opts = {}) {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
  if (!gl) return null;                                     // caller's tokens ground remains
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const sh = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const U = n => gl.getUniformLocation(prog, n);
  const uRes = U("uRes"), uTime = U("uTime"), uMouse = U("uMouse"),
        uNight = U("uNight"), uAct = U("uAct");

  let night = opts.night ? 1 : 0, nightT = night;
  let act = 0, actT = 0;                                    // target + smoothed activity
  let mx = -1e3, my = -1e3, DPR = 1, silkT = 20, dead = false;

  function resize() {
    DPR = Math.min(devicePixelRatio || 1, 2);
    const b = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, b.width * DPR);
    canvas.height = Math.max(1, b.height * DPR);
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  const ro = new ResizeObserver(() => { resize(); if (reduce) draw(0); });
  ro.observe(canvas); resize();

  canvas.addEventListener("pointermove", e => {
    const b = canvas.getBoundingClientRect();
    mx = (e.clientX - b.left) * DPR; my = canvas.height - (e.clientY - b.top) * DPR;
  });
  canvas.addEventListener("pointerleave", () => { mx = -1e3; my = -1e3; });

  function draw(dt) {
    nightT += (night - nightT) * .06;
    actT += (act - actT) * (act > actT ? .10 : .02);        // rises fast, settles slow
    act = Math.max(0, act - dt * .25);                      // activity decays — motion must be re-earned
    silkT += dt * (.06 + .94 * actT);                       // MOTION = MEANING: time flows with activity
    gl.uniform2f(uRes, canvas.width, canvas.height);
    gl.uniform1f(uTime, silkT);
    gl.uniform2f(uMouse, mx, my);
    gl.uniform1f(uNight, reduce ? night : nightT);
    gl.uniform1f(uAct, reduce ? 0 : actT);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  let last = performance.now();
  if (reduce) draw(0);
  else (function loop(now) {
    if (dead) return;
    draw(Math.min(.05, (now - last) / 1000)); last = now;
    requestAnimationFrame(loop);
  })(last);

  return {
    /* raise on REAL events only: generation streaming, replay, meaningful interaction */
    pulse(level = 1) { act = Math.max(act, Math.min(1, level)); if (reduce) draw(0); },
    setNight(n) { night = n ? 1 : 0; if (reduce) draw(0); },
    destroy() { dead = true; ro.disconnect(); },
  };
}
