(() => {
  const card = [...document.querySelectorAll('.predictcard')].find(element =>
    element.closest('section')?.querySelector('.step')?.textContent.trim() === '10');
  if (!card) return;

  const section = card.closest('section');
  section.querySelector('.marginlabel h2').textContent = 'Prueba el modelo';
  section.querySelector('.step').textContent = 'ESTIMADOR';
  section.querySelector('.content > p').innerHTML = 'Ingresa las características de un <strong>distrito censal</strong>. El resultado está expresado en dólares de 1990.';
  document.querySelector('header.hero').insertAdjacentElement('afterend', section);

  card.outerHTML = `
    <form class="prediction-form" id="prediction-form">
      ${field('longitude','Longitud','-122.23','0.01','Rango de California: -124.48 a -114.13.','min="-124.48" max="-114.13"')}
      ${field('latitude','Latitud','37.88','0.01','Rango de California: 32.53 a 42.01.','min="32.53" max="42.01"')}
      ${field('housing_median_age','Antigüedad mediana (años)','41','1','','min="1" max="52"')}
      ${field('median_income_dollars','Ingreso mediano anual (USD)','83252','1','La conversión para el modelo es automática.','min="4999" max="150001"')}
      ${field('total_rooms','Habitaciones totales del distrito','880','1','','min="1"')}
      ${field('total_bedrooms','Dormitorios totales del distrito','129','1','','min="1"')}
      ${field('population','Población del distrito','322','1','','min="1"')}
      ${field('households','Hogares del distrito','126','1','','min="1"')}
      <label class="wide">Proximidad al océano<select name="ocean_proximity" required>
        <option value="NEAR BAY">Cerca de la bahía</option><option value="<1H OCEAN">A menos de 1 hora del océano</option>
        <option value="INLAND">Interior</option><option value="NEAR OCEAN">Cerca del océano</option><option value="ISLAND">Isla</option>
      </select></label>
      <button class="wide" type="submit">Calcular precio estimado</button>
      <div class="prediction-error wide" id="prediction-error" role="alert"></div>
    </form>
    <div class="prediction-output" id="prediction-output" hidden>
      <div class="predictcard"><div><div class="k">Rango orientativo entre árboles</div><div class="prediction-meta" id="prediction-interval"></div></div>
        <div class="result"><div class="k">Precio estimado</div><div class="v" id="prediction-price"></div></div></div>
      <div class="prediction-meta" id="prediction-context"></div><div class="prediction-warning" id="prediction-warning"></div>
      <canvas class="prediction-chart" id="prediction-chart" aria-label="Distribución de precios y estimación"></canvas>
    </div>`;

  const form = document.querySelector('#prediction-form');
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button');
    const error = document.querySelector('#prediction-error');
    error.textContent = '';
    if (!validateRelationships(form, error)) return;
    button.disabled = true; button.textContent = 'Calculando…';
    const payload = Object.fromEntries(new FormData(form));
    payload.median_income = Number(payload.median_income_dollars) / 10000;
    delete payload.median_income_dollars;
    try {
      const response = await fetch('/api/predecir', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'No se pudo calcular la estimación.');
      const money = value => new Intl.NumberFormat('es-MX',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(value);
      document.querySelector('#prediction-price').textContent = money(data.estimate);
      document.querySelector('#prediction-interval').textContent = `${money(data.reference_interval[0])} – ${money(data.reference_interval[1])}. Error absoluto medio del modelo: ${money(data.mae)}.`;
      document.querySelector('#prediction-context').textContent = data.model_context;
      document.querySelector('#prediction-warning').textContent = data.warnings.length ? `Advertencia: ${data.warnings.join('; ')}.` : '';
      document.querySelector('#prediction-output').hidden = false;
      drawHistogram(data.histogram, data.estimate);
    } catch (exception) { error.textContent = exception.message; }
    finally { button.disabled = false; button.textContent = 'Calcular precio estimado'; }
  });

  function field(name,label,value,step,help='',constraints='') {
    return `<label>${label}<input name="${name}" type="number" value="${value}" step="${step}" ${constraints} required>${help?`<span class="field-help">${help}</span>`:''}</label>`;
  }
  function validateRelationships(currentForm,error) {
    if (!currentForm.checkValidity()) { currentForm.reportValidity(); return false; }
    const value = name => Number(currentForm.elements[name].value);
    const problems = [];
    if (value('total_bedrooms') > value('total_rooms')) problems.push('Los dormitorios no pueden superar las habitaciones totales.');
    if (value('households') > value('population')) problems.push('Los hogares no pueden superar la población.');
    if (problems.length) { error.textContent = problems.join(' '); return false; }
    return true;
  }
  function drawHistogram(hist,estimate) {
    const canvas=document.querySelector('#prediction-chart'), dpr=window.devicePixelRatio||1, w=canvas.clientWidth, h=canvas.clientHeight;
    canvas.width=w*dpr; canvas.height=h*dpr; const c=canvas.getContext('2d'); c.scale(dpr,dpr); c.clearRect(0,0,w,h);
    const pad={l:42,r:18,t:24,b:38}, pw=w-pad.l-pad.r, ph=h-pad.t-pad.b, max=Math.max(...hist.counts);
    c.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ocean-soft');
    hist.counts.forEach((n,i)=>{const bw=pw/hist.counts.length; c.fillRect(pad.l+i*bw,pad.t+ph-(n/max)*ph,Math.max(1,bw-1),(n/max)*ph);});
    const min=hist.edges[0], maxX=hist.edges.at(-1), x=pad.l+Math.max(0,Math.min(1,(estimate-min)/(maxX-min)))*pw;
    c.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--gold'); c.lineWidth=3; c.beginPath(); c.moveTo(x,pad.t); c.lineTo(x,pad.t+ph); c.stroke();
    c.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--ink-soft'); c.font='12px IBM Plex Sans'; c.fillText('Distribución de precios de entrenamiento',pad.l,15); c.fillText('$'+Math.round(min/1000)+'k',pad.l,pad.t+ph+22); c.fillText('$'+Math.round(maxX/1000)+'k',pad.l+pw-34,pad.t+ph+22); c.fillText('Estimación',Math.min(x+5,w-75),pad.t+14);
  }
})();
