(() => {
  const card = [...document.querySelectorAll('.predictcard')].find(element =>
    element.closest('section')?.querySelector('.step')?.textContent.trim() === '10');
  if (!card) return;

  const section = card.closest('section');
  section.querySelector('.marginlabel h2').textContent = 'Prueba el modelo';
  section.querySelector('.step').textContent = 'ESTIMADOR';
  section.querySelector('.content > p').innerHTML = 'Ingresa las características de un <strong>distrito censal</strong>, o haz clic en el mapa para fijar la ubicación. El resultado está expresado en dólares de 1990.';
  document.querySelector('header.hero').insertAdjacentElement('afterend', section);

  card.outerHTML = `
    <div class="map-panel" id="map-panel" hidden>
      <div class="ca-map" id="ca-map"></div>
      <div class="map-toolbar">
        <button type="button" id="toggle-surface">Ver superficie de precios del modelo</button>
        <button type="button" id="load-example">Cargar un distrito real</button>
        <span class="map-hint">Clic en el mapa para fijar la ubicación · arrastra el pin para ajustar</span>
      </div>
      <div class="map-legend" id="map-legend" hidden>
        <span id="legend-lo"></span><span class="legend-bar"></span><span id="legend-hi"></span>
        <span class="legend-title">Precio mediano estimado por distrito</span>
      </div>
    </div>
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
      <div class="predictcard"><div><div class="k">Rango orientativo (P10–P90)</div><div class="prediction-meta" id="prediction-interval"></div></div>
        <div class="result"><div class="k">Precio estimado</div><div class="v" id="prediction-price"></div></div></div>
      <div class="prediction-real" id="prediction-real" hidden></div>
      <div class="prediction-meta" id="prediction-context"></div><div class="prediction-warning" id="prediction-warning"></div>
      <canvas class="prediction-chart" id="prediction-chart" aria-label="Distribución de precios y estimación"></canvas>
    </div>`;

  const form = document.querySelector('#prediction-form');
  const money = value => new Intl.NumberFormat('es-MX',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(value);

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button[type="submit"]');
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
      document.querySelector('#prediction-price').textContent = money(data.estimate);
      document.querySelector('#prediction-interval').textContent = `${money(data.reference_interval[0])} – ${money(data.reference_interval[1])}. Error absoluto medio del modelo: ${money(data.mae)}.`;
      document.querySelector('#prediction-context').textContent = data.model_context;
      document.querySelector('#prediction-warning').textContent = data.warnings.length ? `Advertencia: ${data.warnings.join('; ')}.` : '';
      renderRealValue(data.estimate);
      document.querySelector('#prediction-output').hidden = false;
      drawHistogram(data.histogram, data.estimate);
    } catch (exception) { error.textContent = exception.message; }
    finally { button.disabled = false; button.textContent = 'Calcular precio estimado'; }
  });

  function renderRealValue(estimate) {
    const box = document.querySelector('#prediction-real');
    const real = Number(form.dataset.realValue);
    if (!real) { box.hidden = true; return; }
    const diff = estimate - real;
    const pct = Math.abs(diff / real) * 100;
    box.innerHTML = `Distrito real del censo · valor observado <strong>${money(real)}</strong>. ` +
      `El modelo ${diff >= 0 ? 'sobrestima' : 'subestima'} por <strong>${money(Math.abs(diff))}</strong> (${pct.toFixed(1)}%).`;
    box.hidden = false;
  }

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

  // ---------- Mapa interactivo de California (mejora progresiva) ----------
  enhanceWithMap().catch(() => {});

  function loadLeaflet() {
    if (window.L) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css';
      document.head.appendChild(css);
      const script = document.createElement('script');
      script.src = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }

  async function enhanceWithMap() {
    const response = await fetch('/mapa.json');
    if (!response.ok) return;
    const data = await response.json();
    await loadLeaflet();

    const panel = document.querySelector('#map-panel');
    panel.hidden = false;
    const map = L.map('ca-map', { preferCanvas: true, scrollWheelZoom: false, zoomControl: true })
      .setView([37.1, -119.4], 6);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap', maxZoom: 12,
    }).addTo(map);
    setTimeout(() => map.invalidateSize(), 250);

    const lonInput = form.elements.longitude;
    const latInput = form.elements.latitude;
    const proxInput = form.elements.ocean_proximity;
    let marker = null;
    let debounce = null;

    function livePredict() {
      clearTimeout(debounce);
      debounce = setTimeout(() => {
        if (form.checkValidity()) form.requestSubmit();
      }, 250);
    }

    function nearestCell(lat, lon) {
      let best = null, bestDist = Infinity;
      for (const cell of data.cells) {
        const d = (cell[0] - lat) ** 2 + (cell[1] - lon) ** 2;
        if (d < bestDist) { bestDist = d; best = cell; }
      }
      return best;
    }

    function setPoint(lat, lon, { snapProx = true, clearReal = true } = {}) {
      lat = Number(lat.toFixed(2));
      lon = Number(lon.toFixed(2));
      latInput.value = lat;
      lonInput.value = lon;
      if (!marker) {
        marker = L.marker([lat, lon], { draggable: true }).addTo(map);
        marker.on('dragend', () => {
          const pos = marker.getLatLng();
          setPoint(pos.lat, pos.lng);
        });
      } else {
        marker.setLatLng([lat, lon]);
      }
      if (snapProx) {
        const cell = nearestCell(lat, lon);
        if (cell) proxInput.value = data.ocean_labels[cell[3]];
      }
      if (clearReal) delete form.dataset.realValue;
      livePredict();
    }

    map.on('click', event => setPoint(event.latlng.lat, event.latlng.lng));
    setPoint(Number(latInput.value), Number(lonInput.value), { snapProx: false, clearReal: false });

    // Superficie de precios --------------------------------------------------
    const [lo, hi] = data.price_domain;
    const legend = document.querySelector('#map-legend');
    const toggleButton = document.querySelector('#toggle-surface');
    let surface = null;

    function colorFor(price) {
      const t = Math.max(0, Math.min(1, (price - lo) / (hi - lo)));
      const pale = [244, 233, 214], deep = [16, 74, 84];
      const mix = pale.map((v, i) => Math.round(v + (deep[i] - v) * t));
      return `rgb(${mix[0]},${mix[1]},${mix[2]})`;
    }

    document.querySelector('#legend-lo').textContent = money(lo);
    document.querySelector('#legend-hi').textContent = money(hi);
    document.querySelector('.legend-bar').style.background =
      `linear-gradient(90deg, ${colorFor(lo)}, ${colorFor((lo + hi) / 2)}, ${colorFor(hi)})`;

    toggleButton.addEventListener('click', () => {
      if (surface) {
        map.removeLayer(surface);
        surface = null;
        legend.hidden = true;
        toggleButton.textContent = 'Ver superficie de precios del modelo';
        return;
      }
      const half = data.grid_step / 2;
      surface = L.layerGroup(data.cells.map(cell =>
        L.rectangle(
          [[cell[0] - half, cell[1] - half], [cell[0] + half, cell[1] + half]],
          { stroke: false, fillColor: colorFor(cell[2]), fillOpacity: 0.6 },
        ).bindTooltip(money(cell[2]), { sticky: true }),
      )).addTo(map);
      legend.hidden = false;
      toggleButton.textContent = 'Ocultar superficie de precios';
    });

    // Cargar un distrito real ---------------------------------------------------
    let exampleIndex = 0;
    document.querySelector('#load-example').addEventListener('click', () => {
      const example = data.examples[exampleIndex % data.examples.length];
      exampleIndex += 1;
      form.elements.housing_median_age.value = Math.round(example.housing_median_age);
      form.elements.total_rooms.value = Math.round(example.total_rooms);
      form.elements.total_bedrooms.value = Math.round(example.total_bedrooms);
      form.elements.population.value = Math.round(example.population);
      form.elements.households.value = Math.round(example.households);
      form.elements.median_income_dollars.value = Math.round(example.median_income * 10000);
      proxInput.value = example.ocean_proximity;
      map.setView([example.latitude, example.longitude], 8);
      setPoint(example.latitude, example.longitude, { snapProx: false, clearReal: false });
      form.dataset.realValue = example.real;
      livePredict();
    });
  }
})();
