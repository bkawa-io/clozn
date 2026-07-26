const ROUTES={"":"Runs",runs:"Runs",model:"Model",models:"Models",compare:"Compare",behavior:"Behavior",observatory:"Observatory"};
function routeName(){const raw=location.hash.replace(/^#\/?/,"").split(/[/?]/)[0];return ROUTES[raw]||raw||"Studio"}
function makeButton(label,title,onClick){const el=document.createElement("button");el.type="button";el.className="workspace-toggle";el.textContent=label;el.title=title;el.addEventListener("click",onClick);return el}
export function mountWorkspace(shell=document.querySelector(".shell"),options={}){
 if(!shell)return{destroy(){},setInspector(){},setTimeline(){}};
 if(shell.dataset.workspaceMounted==="true")return shell.__cloznWorkspace;
 const compact=matchMedia("(max-width:1179px)").matches;
 shell.dataset.workspaceMounted="true";shell.classList.add("clozn-workspace");shell.dataset.inspector=options.inspector==null?(compact?"closed":"open"):(options.inspector===false?"closed":"open");shell.dataset.timeline=options.timeline===false?"closed":"open";
 const header=shell.querySelector("header.bar");
 const inspector=document.createElement("aside");inspector.className="workspace-inspector";inspector.setAttribute("aria-label","Inspector");inspector.innerHTML='<div class="workspace-region-head"><span>Inspector</span><span class="spacer"></span></div><div class="workspace-region-body"><div class="workspace-empty">Select evidence in the viewport<br>to inspect its recorded values.</div></div>';
 const timeline=document.createElement("section");timeline.className="workspace-timeline";timeline.setAttribute("aria-label","Replay timeline");timeline.innerHTML='<div class="workspace-region-head"><span>Replay</span><span class="spacer"></span></div><div class="workspace-region-body"><div class="workspace-empty">A route may mount ordered inference events here.</div></div>';
 shell.append(inspector,timeline);
 const ib=makeButton("I","Toggle inspector (I)",()=>shell.dataset.inspector=shell.dataset.inspector==="closed"?"open":"closed");
 const tb=makeButton("T","Toggle timeline (T)",()=>shell.dataset.timeline=shell.dataset.timeline==="closed"?"open":"closed");
 const closeInspector=makeButton("X","Close inspector",()=>shell.dataset.inspector="closed");closeInspector.classList.add("workspace-close");closeInspector.setAttribute("aria-label","Close inspector");inspector.querySelector(".workspace-region-head").append(closeInspector);
 let cleanupRoute=()=>{};
 if(header){const spacer=header.querySelector(".spacer");const route=document.createElement("span");route.className="workspace-route";route.style.cssText="font-family:var(--voice-machine);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-faint)";route.textContent=routeName();header.insertBefore(route,spacer||header.firstChild);header.insertBefore(ib,spacer||null);header.insertBefore(tb,spacer||null);const update=()=>route.textContent=routeName();window.addEventListener("hashchange",update);cleanupRoute=()=>window.removeEventListener("hashchange",update)}
 function onKey(e){if(e.defaultPrevented||e.metaKey||e.ctrlKey||e.altKey)return;const t=e.target;if(t&&/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))return;if(e.key.toLowerCase()==="i")ib.click();if(e.key.toLowerCase()==="t")tb.click();if(e.key==="Escape"&&innerWidth<1180)shell.dataset.inspector="closed"}
 window.addEventListener("keydown",onKey);
 const api={inspectorBody:inspector.querySelector(".workspace-region-body"),timelineBody:timeline.querySelector(".workspace-region-body"),setInspector(content){this.inspectorBody.replaceChildren();content instanceof Node?this.inspectorBody.append(content):this.inspectorBody.innerHTML=String(content||"")},setTimeline(content){this.timelineBody.replaceChildren();content instanceof Node?this.timelineBody.append(content):this.timelineBody.innerHTML=String(content||"")},showInspector(show=true){shell.dataset.inspector=show?"open":"closed"},showTimeline(show=true){shell.dataset.timeline=show?"open":"closed"},destroy(){window.removeEventListener("keydown",onKey);cleanupRoute();inspector.remove();timeline.remove();ib.remove();tb.remove();shell.classList.remove("clozn-workspace");delete shell.dataset.workspaceMounted;delete shell.__cloznWorkspace}};
 shell.__cloznWorkspace=api;window.cloznWorkspace=api;return api;
}
