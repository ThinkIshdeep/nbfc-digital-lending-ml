const state = {summary:null, trends:null, mix:null, models:null, segments:null, anomalies:null};
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const fmt = (n, digits=0) => Number(n).toLocaleString('en-IN',{maximumFractionDigits:digits,minimumFractionDigits:digits});
const crore = (n) => `₹${fmt(n)} Cr`;

async function api(path, options){
  const r = await fetch(path, options);
  if(!r.ok) throw new Error((await r.text()) || `${r.status}`);
  return r.json();
}

function navigate(view){
  $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  $$('.view').forEach(v => v.classList.toggle('active', v.id === `view-${view}`));
  const titles = {overview:'Digital Lending Overview',predict:'Analytical Scenario', 'model-lab':'Model Lab',segments:'Behaviour Segments',data:'Data Foundation'};
  $('#pageTitle').textContent = titles[view] || 'NBFC Intelligence';
  $('#sidebar').classList.remove('open');
}
$$('.nav-item').forEach(b => b.addEventListener('click',()=>navigate(b.dataset.view)));
$('#mobileMenu').addEventListener('click',()=>$('#sidebar').classList.toggle('open'));

function metricCard(label,value,meta){return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-meta">${meta}</div></div>`}
function setReady(ready){const el=$('#readyPill');el.textContent=ready?'● Models ready':'● Service unavailable';el.style.color=ready?'#147a5a':'#a33d3d'}

function lineChart(svgId, labels, values){
  const svg=$(svgId); const W=760,H=280,L=54,R=22,T=20,B=40;
  const max=Math.max(...values)*1.12 || 1;
  const x=(i)=>L+(i/(Math.max(values.length-1,1)))*(W-L-R);
  const y=(v)=>T+(1-v/max)*(H-T-B);
  const grid=[0,.25,.5,.75,1].map(p=>{const yy=y(max*p);return `<line class="chart-grid" x1="${L}" y1="${yy}" x2="${W-R}" y2="${yy}"/><text class="chart-text" x="${L-8}" y="${yy+4}" text-anchor="end">${fmt(max*p,0)}</text>`}).join('');
  const points=values.map((v,i)=>`${x(i)},${y(v)}`).join(' ');
  const dots=values.map((v,i)=>`<circle class="chart-dot" cx="${x(i)}" cy="${y(v)}" r="4"></circle><text class="chart-text" x="${x(i)}" y="${H-12}" text-anchor="middle">${labels[i].replace('As on ','')}</text>`).join('');
  svg.innerHTML=`<g>${grid}<polyline class="chart-line" points="${points}"></polyline>${dots}</g>`;
}

function renderOverview(){
  const p=state.summary;
  $('#summaryCards').innerHTML=[
    metricCard('Core cells',fmt(p.core_cells),'61 visible respondent rows × 5 periods'),
    metricCard('ML transitions',fmt(p.transition_rows),'51 respondent groups with complete transitions'),
    metricCard('Latest disbursement',crore(p.latest_disbursement_crore),'Observed aggregate at final snapshot'),
    metricCard('Latest avg. ticket',`₹${fmt(p.latest_weighted_average_ticket_rupees)}`,'Weighted from complete amount/count cells')
  ].join('');
  const t=state.trends.periods;
  lineChart('#disbursementChart',t.map(x=>x.period),t.map(x=>x.amount_total_crore));
  lineChart('#loanChart',t.map(x=>x.period),t.map(x=>x.loan_count_total));
  const mix=state.mix.slice(0,6); const mx=Math.max(...mix.map(x=>x.nbfc_amount_crore));
  $('#productMix').innerHTML=mix.map(x=>`<div class="mix-row"><span class="mix-name">${x.loan_purpose}</span><span class="mix-value">${crore(x.nbfc_amount_crore)}</span><div class="mix-track"><div class="mix-fill" style="width:${(x.nbfc_amount_crore/mx*100).toFixed(1)}%"></div></div></div>`).join('');
  const c=state.models.classification.find(x=>x.model===state.models.best_classifier); const r=state.models.regression.find(x=>x.model===state.models.best_regressor); const hold=state.models.temporal;
  $('#healthGrid').innerHTML=`<div class="health-item"><span>Grouped CV · classifier F1</span><strong>${c.f1.toFixed(3)}</strong></div><div class="health-item"><span>Grouped CV · regressor R²</span><strong>${r.r2.toFixed(3)}</strong></div><div class="health-item"><span>Time holdout · classifier F1</span><strong class="warning-text">${hold.classifier.f1.toFixed(3)}</strong></div><div class="health-item"><span>Time holdout · regressor R²</span><strong class="warning-text">${hold.regressor.r2.toFixed(3)}</strong></div>`;
}

function renderModels(){
  const m=state.models;
  $('#classificationTable').innerHTML=`<div class="table-scroll"><table class="model-table"><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>${m.classification.map(x=>`<tr class="${x.model===m.best_classifier?'best-row':''}"><td>${x.model} ${x.model===m.best_classifier?'<span class="best-chip">SELECTED</span>':''}</td><td>${(x.accuracy*100).toFixed(1)}%</td><td>${(x.precision*100).toFixed(1)}%</td><td>${(x.recall*100).toFixed(1)}%</td><td><strong>${x.f1.toFixed(3)}</strong></td></tr>`).join('')}</tbody></table></div>`;
  $('#regressionTable').innerHTML=`<div class="table-scroll"><table class="model-table"><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R²</th></tr></thead><tbody>${m.regression.map(x=>`<tr class="${x.model===m.best_regressor?'best-row':''}"><td>${x.model} ${x.model===m.best_regressor?'<span class="best-chip">SELECTED</span>':''}</td><td>₹${fmt(x.MAE_crore,1)} Cr</td><td>₹${fmt(x.RMSE_crore,1)} Cr</td><td><strong>${x.R2.toFixed(3)}</strong></td></tr>`).join('')}</tbody></table></div>`;
  $('#temporalTable').innerHTML=`<div class="table-scroll"><table class="model-table"><thead><tr><th>Task</th><th>Baseline / Model</th><th>Primary</th><th>Holdout</th></tr></thead><tbody><tr><td>Direction</td><td>Majority baseline</td><td>F1 0.603</td><td>F1 ${(m.temporal.classifier_baseline.f1).toFixed(3)}</td></tr><tr class="best-row"><td>Direction</td><td>${m.best_classifier}</td><td>F1 ${m.classification.find(x=>x.model===m.best_classifier).f1.toFixed(3)}</td><td>F1 <strong>${m.temporal.classifier.f1.toFixed(3)}</strong></td></tr><tr><td>Amount</td><td>Persistence baseline</td><td>RMSE ₹${fmt(m.temporal.regressor_baseline.rmse_crore,0)} Cr</td><td>R² ${m.temporal.regressor_baseline.r2.toFixed(3)}</td></tr><tr class="best-row"><td>Amount</td><td>${m.best_regressor}</td><td>RMSE ₹${fmt(m.regression.find(x=>x.model===m.best_regressor).RMSE_crore,0)} Cr</td><td>R² <strong>${m.temporal.regressor.r2.toFixed(3)}</strong></td></tr></tbody></table></div><p class="temporal-note">The holdout is the final observed transition into 31 Dec 2020. Negative R² means the model did not generalise well to that period.</p>`;
  const fi=m.feature_importance; const max=Math.max(...fi.map(x=>x.importance));
  $('#featureBars').innerHTML=fi.map(x=>`<div class="feature-row"><div class="feature-top"><span>${x.feature}</span><span>${(x.importance*100).toFixed(1)}%</span></div><div class="feature-track"><div class="feature-fill" style="width:${(x.importance/max*100).toFixed(1)}%"></div></div></div>`).join('');
}

function renderSegments(){
  $('#segmentCards').innerHTML=state.segments.map(s=>`<article class="segment-card"><span class="segment-no">CLUSTER ${s.cluster}</span><h3>${s.label}</h3><p>${s.description}</p><div class="segment-stats"><div class="segment-stat"><span>NBFCs</span><strong>${s.nbfc_count}</strong></div><div class="segment-stat"><span>Mean disbursement</span><strong>₹${fmt(s.mean_disbursement_crore,1)} Cr</strong></div><div class="segment-stat"><span>Mean loans</span><strong>${fmt(s.mean_loan_count)}</strong></div><div class="segment-stat"><span>Mean ticket</span><strong>₹${fmt(s.mean_average_ticket_rupees)}</strong></div></div></article>`).join('');
  $('#anomalyTable').innerHTML=`<div class="table-scroll"><table class="model-table"><thead><tr><th>Respondent</th><th>Disbursement</th><th>Loans</th><th>Ticket</th><th>Signal</th></tr></thead><tbody>${state.anomalies.map(x=>`<tr><td>${x.nbfc_respondent_id}</td><td>₹${fmt(x.digital_disbursement_crore,1)} Cr</td><td>${fmt(x.digital_loan_count)}</td><td>₹${fmt(x.average_ticket_rupees)}</td><td>${x.anomaly_flag===-1?'<span class="warning-text">Potentially unusual</span>':'Within pattern'}</td></tr>`).join('')}</tbody></table></div>`;
}

function renderPrediction(r){
  const increased=r.next_disbursement_increased;
  $('#predictionResult').innerHTML=`<div class="result-top"><div><h3>Direction</h3><div class="result-direction ${increased?'up':'down'}">${increased?'Increase':'Decrease / same'}</div><p class="muted">Classifier probability of increase: ${r.increase_probability==null?'n/a':(r.increase_probability*100).toFixed(1)+'%'}</p></div><span class="badge">${r.classifier}</span></div><div class="result-grid"><div class="result-cell"><span>Estimated next-period amount</span><strong>₹${fmt(r.predicted_next_disbursement_crore,2)} Cr</strong></div><div class="result-cell"><span>Behaviour cluster</span><strong>Cluster ${r.cluster}</strong></div><div class="result-cell"><span>Anomaly signal</span><strong>${r.anomaly_flag==='potentially_unusual'?'Review':'Within observed pattern'}</strong></div></div><div class="callout">Model outputs are analytical estimates from aggregate RBI data. They should not be used to approve, decline, price or limit an individual borrower's credit.</div>`;
}

$('#predictForm').addEventListener('submit', async (e)=>{
  e.preventDefault(); const btn=e.submitter; btn.disabled=true; btn.textContent='Running…';
  try{const r=await api('/v1/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({current_disbursement_crore:Number($('#disbursement').value),current_loan_count:Number($('#loanCount').value),current_average_ticket_rupees:Number($('#avgTicket').value),period_index:Number($('#periodIndex').value)})});renderPrediction(r)}catch(err){$('#predictionResult').innerHTML=`<div class="empty-state"><h3>Request failed</h3><p>${err.message}</p></div>`}finally{btn.disabled=false;btn.innerHTML='Run model <span>→</span>'}}
);

async function load(){
  try{
    setReady(true);
    [state.summary,state.trends,state.mix,state.models,state.segments,state.anomalies]=await Promise.all([
      api('/v1/dashboard/summary'),api('/v1/dashboard/trends'),api('/v1/dashboard/product-mix'),api('/v1/dashboard/model-lab'),api('/v1/dashboard/segments'),api('/v1/dashboard/anomalies')
    ]);
    renderOverview();renderModels();renderSegments();
    const ready=await api('/ready'); setReady(ready.status==='ready');
  }catch(e){console.error(e);setReady(false);document.querySelectorAll('.panel').forEach(p=>{if(!p.querySelector('form') && !p.querySelector('.schema-list')) return})}
}
load();
