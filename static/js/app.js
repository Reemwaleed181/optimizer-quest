// ============================================================
//  app.js - Main controller: wires backend + UI
// ============================================================

window.activeOpt = 'Adam';

const API_BASE = window.location.protocol === 'file:' ? null : window.location.origin;

const DATASET_INFO = {
  mnist: {
    label: 'MNIST',
    help: 'Handwritten digits; easiest benchmark for optimizer comparison.',
    desc: 'Classic handwritten digit dataset with 10 balanced digit classes. Best for quick optimizer behavior demonstrations.',
    task: 'Digit recognition'
  },
  fashionmnist: {
    label: 'Fashion-MNIST',
    help: 'Clothing images; harder than MNIST and useful for testing generalization.',
    desc: 'Grayscale clothing dataset with 10 apparel classes. It is visually more ambiguous than digits, so validation behavior is more informative.',
    task: 'Apparel classification'
  },
  kmnist: {
    label: 'KMNIST',
    help: 'Japanese Kuzushiji characters; challenging shape recognition benchmark.',
    desc: 'Kuzushiji character dataset with 10 classes. It stresses shape discrimination and often benefits from convolutional models.',
    task: 'Character recognition'
  }
};

const MODEL_INFO = {
  tinymlp: {
    label: 'Tiny MLP',
    help: 'Fastest architecture; choose it for quick baseline runs, not best image accuracy.',
    desc: 'Flattened image input with two dense layers. Good for teaching why spatial structure matters.'
  },
  simplecnn: {
    label: 'Simple CNN',
    help: 'Balanced CNN baseline; best default for all three datasets.',
    desc: 'Two convolution blocks plus a compact classifier. Good balance of speed, accuracy, and interpretability.'
  },
  deepercnn: {
    label: 'Deeper CNN',
    help: 'More expressive CNN; choose it for harder datasets or stronger final accuracy.',
    desc: 'Four convolution layers with batch normalization and dropout. Better capacity, but slower per epoch.'
  }
};

const BATCH_INFO = {
  32: 'Smaller batch: noisier gradient updates, often better for teaching optimizer movement, but slower per epoch.',
  64: 'Balanced batch: stable updates with reasonable speed. Good default for this interactive demo.',
  128: 'Larger batch: faster epoch throughput, smoother gradients, but sometimes less responsive optimizer behavior.'
};

const EPOCH_INFO = {
  2: 'Fast demo run: enough to verify the pipeline and compare early optimizer behavior.',
  5: 'Short experiment: better for seeing convergence trends without waiting too long.',
  10: 'Stronger comparison: useful when judging validation accuracy and generalization.',
  20: 'Longer run: best for final accuracy comparisons, but slower on CPU.'
};

let running = false;
let benchmarkRunning = false;
let maxEpochs = 2;
let bestLoss = Infinity;
let scores = {};
let runHistory = [];
let pollTimer = null;
let elapsedTimer = null;
let localStartTime = null;
let lastRenderedEpoch = 0;

// -- Startup --------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
  initCharts(window.activeOpt);
  initArena();
  initTheme();
  updateInfoPanel(window.activeOpt);
  updateExperimentContext();
  updateExperimentDraft();
  bindEvents();
  pollTrainingStatus(false);
});

// -- Event Binding --------------------------------------------
function bindEvents() {
  enhanceSettingSelects();

  document.querySelectorAll('.opt-item').forEach(el => {
    el.addEventListener('click', () => {
      if (running) return;
      selectOptimizer(el.dataset.opt);
    });
  });

  document.getElementById('runBtn')?.addEventListener('click', toggleTraining);
  document.getElementById('startBtn')?.addEventListener('click', toggleTraining);
  document.getElementById('themeToggle')?.addEventListener('click', toggleTheme);
  document.getElementById('benchmarkBtn')?.addEventListener('click', startOptimizerBenchmark);

  ['setLR', 'setBatch', 'setEpochs', 'setDataset', 'setModel', 'setLossFn'].forEach(id => {
    document.getElementById(id)?.addEventListener('change', updateExperimentDraft);
  });

  document.getElementById('setDataset')?.addEventListener('change', updateExperimentContext);
  document.getElementById('setModel')?.addEventListener('change', updateExperimentContext);
  document.getElementById('setLossFn')?.addEventListener('change', updateLossContext);
  document.getElementById('setLR')?.addEventListener('change', updateLearningRateHint);
  document.getElementById('exportHistoryBtn')?.addEventListener('click', exportRunHistoryCsv);
  document.getElementById('resetHistoryBtn')?.addEventListener('click', resetRunHistory);
}

function initTheme() {
  const saved = window.localStorage.getItem('optimizerQuestTheme') || 'dark';
  document.body.dataset.theme = saved;
  updateThemeButton(saved);
}

function toggleTheme() {
  const current = document.body.dataset.theme === 'light' ? 'light' : 'dark';
  const next = current === 'light' ? 'dark' : 'light';
  document.body.dataset.theme = next;
  window.localStorage.setItem('optimizerQuestTheme', next);
  updateThemeButton(next);
}

function updateThemeButton(theme) {
  const btn = document.getElementById('themeToggle');
  if (btn) btn.textContent = theme === 'light' ? 'Dark' : 'Light';
}

function enhanceSettingSelects() {
  const ids = ['setDataset', 'setModel', 'setLR', 'setBatch', 'setEpochs', 'setLossFn'];

  ids.forEach(id => {
    const select = document.getElementById(id);
    if (!select || select.dataset.enhanced === 'true') return;

    select.dataset.enhanced = 'true';
    select.classList.add('native-select-hidden');

    const shell = document.createElement('div');
    shell.className = 'custom-select';

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'custom-select-trigger';
    button.textContent = select.options[select.selectedIndex]?.textContent || 'Select';

    const menu = document.createElement('div');
    menu.className = 'custom-select-menu';
    menu.hidden = true;

    Array.from(select.options).forEach(option => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'custom-select-option';
      item.dataset.value = option.value;
      item.textContent = option.textContent;

      item.addEventListener('mouseenter', () => {
        const [title, text] = getSettingOptionInfo(id, option.value, option.textContent);
        showOptionTooltip(item, title, text);
      });
      item.addEventListener('focus', () => {
        const [title, text] = getSettingOptionInfo(id, option.value, option.textContent);
        showOptionTooltip(item, title, text);
      });

      item.addEventListener('click', () => {
        select.value = option.value;
        button.textContent = option.textContent;
        menu.hidden = true;
        hideOptionTooltip();
        select.dispatchEvent(new Event('change', { bubbles: true }));
      });

      menu.appendChild(item);
    });

    button.addEventListener('click', event => {
      event.stopPropagation();
      closeCustomSelects(shell);
      menu.hidden = !menu.hidden;
      const [title, text] = getSettingOptionInfo(
        id,
        select.value,
        select.options[select.selectedIndex]?.textContent || ''
      );
      if (menu.hidden) hideOptionTooltip();
      else showOptionTooltip(button, title, text);
    });

    menu.addEventListener('mouseleave', hideOptionTooltip);
    shell.appendChild(button);
    shell.appendChild(menu);
    select.insertAdjacentElement('afterend', shell);
  });

  if (!window.customSelectCloseBound) {
    window.customSelectCloseBound = true;
    document.addEventListener('click', () => {
      closeCustomSelects();
      hideOptionTooltip();
    });
  }
}

function closeCustomSelects(except = null) {
  document.querySelectorAll('.custom-select').forEach(shell => {
    if (shell === except) return;
    const menu = shell.querySelector('.custom-select-menu');
    if (menu) menu.hidden = true;
  });
}

function optionTooltip() {
  let tooltip = document.getElementById('optionTooltip');
  if (tooltip) return tooltip;

  tooltip = document.createElement('div');
  tooltip.id = 'optionTooltip';
  tooltip.className = 'option-tooltip';
  tooltip.hidden = true;
  tooltip.innerHTML = '<strong></strong><p></p>';
  document.body.appendChild(tooltip);
  return tooltip;
}

function showOptionTooltip(anchor, title, text) {
  const tooltip = optionTooltip();
  tooltip.querySelector('strong').textContent = title;
  tooltip.querySelector('p').textContent = text;

  const rect = anchor.getBoundingClientRect();
  const tooltipWidth = 260;
  const canShowRight = window.innerWidth - rect.right > tooltipWidth + 18;
  const left = canShowRight ? rect.right + 10 : Math.max(10, rect.left - tooltipWidth - 10);
  const top = Math.min(window.innerHeight - 130, Math.max(10, rect.top - 6));

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.hidden = false;
}

function hideOptionTooltip() {
  const tooltip = document.getElementById('optionTooltip');
  if (tooltip) tooltip.hidden = true;
}

function getSettingOptionInfo(selectId, value, label) {
  if (selectId === 'setDataset') {
    const info = DATASET_INFO[value] || DATASET_INFO.mnist;
    return [`Dataset: ${info.label}`, `${info.help} ${info.desc}`];
  }

  if (selectId === 'setModel') {
    const info = MODEL_INFO[value] || MODEL_INFO.simplecnn;
    return [`Model: ${info.label}`, `${info.help} ${info.desc}`];
  }

  if (selectId === 'setLR') {
    return [`Learning rate: ${label}`, learningRateText(Number(value))];
  }

  if (selectId === 'setBatch') {
    return [`Batch size: ${label}`, BATCH_INFO[value] || BATCH_INFO[64]];
  }

  if (selectId === 'setEpochs') {
    return [`Epochs: ${label}`, EPOCH_INFO[value] || EPOCH_INFO[2]];
  }

  if (selectId === 'setLossFn') {
    const info = LOSS_CODE[value] || LOSS_CODE.crossentropy;
    return [`Loss: ${info.label}`, info.help];
  }

  return [label, 'Choose this option to configure the experiment.'];
}

// -- Optimizer Selection --------------------------------------
function selectOptimizer(name) {
  window.activeOpt = name;

  document.querySelectorAll('.opt-item').forEach(el => {
    el.classList.toggle('active', el.dataset.opt === name);
  });

  updateInfoPanel(name);
  updateRadarActive(name);
  updateExperimentDraft();
  renderTradeoffs(name);
}

function updateInfoPanel(name) {
  const info = OPT_INFO[name];
  if (!info) return;

  setText('optDescription', info.description);
  setText('optFormula', info.formula);
  setText('perfConv', info.speed);
  updateOptimizerLink(name);

  const hpEl = document.getElementById('optHyperparams');
  if (hpEl) {
    hpEl.innerHTML = info.hyperparameters
      .map(hp => `
        <span class="opt-hyperparam" title="${hp.desc}">
          <span class="hp-name">${hp.name}</span>
          <span class="hp-val">${hp.default}</span>
        </span>
      `)
      .join('');
  }

  const useEl = document.getElementById('optUseCases');
  if (useEl) {
    useEl.innerHTML = info.useCases
      .map(item => `
        <div class="opt-usecase">
          <span class="opt-usecase-dot"></span>
          <span>${item}</span>
        </div>
      `)
      .join('');
  }

}

// -- Training -------------------------------------------------
function toggleTraining() {
  if (running) return;
  startTraining();
}

async function startTraining() {
  if (!API_BASE) {
    alert('Training requires the Flask server. Open http://127.0.0.1:5000/ instead of the HTML file.');
    return;
  }

  const lr = parseFloat(document.getElementById('setLR').value);
  const batchSz = parseInt(document.getElementById('setBatch').value, 10) || 64;
  const dataset = document.getElementById('setDataset')?.value || 'mnist';
  const model = document.getElementById('setModel')?.value || 'simplecnn';
  const lossFunction = document.getElementById('setLossFn')?.value || 'crossentropy';
  maxEpochs = parseInt(document.getElementById('setEpochs').value, 10) || 2;

  try {
    await runTrainingOnce({
      optimizer: window.activeOpt,
      dataset,
      model,
      lossFunction,
      lr,
      epochs: maxEpochs,
      batchSz
    });
  } catch (err) {
    console.error('Training error:', err);
    alert('Training failed: Could not reach the Flask API. Make sure app.py is still running in the terminal, then refresh http://127.0.0.1:5000/.');
    finishTrainingUi('Ready');
  }
}

async function startOptimizerBenchmark() {
  if (!API_BASE) {
    alert('Benchmarking requires the Flask server. Open http://127.0.0.1:5000/ instead of the HTML file.');
    return;
  }
  if (running || benchmarkRunning) return;

  benchmarkRunning = true;
  showBenchmarkPlan();

  const plan = benchmarkProtocol();
  setBenchmarkStatus('Preparing fair benchmark settings...');
  setControlValue('setDataset', plan.dataset);
  setControlValue('setModel', plan.model);
  setControlValue('setLossFn', plan.lossFunction);
  setControlValue('setBatch', String(plan.batchSz));
  setControlValue('setEpochs', String(plan.epochs));
  updateExperimentContext();

  try {
    for (let i = 0; i < plan.optimizers.length; i++) {
      const item = plan.optimizers[i];
      setBenchmarkStatus(`Run ${i + 1}/${plan.optimizers.length}: training ${item.name} with lr=${item.lr}`);
      selectOptimizer(item.name);
      setControlValue('setLR', String(item.lr));
      updateExperimentDraft();

      await runTrainingOnce({
        optimizer: item.name,
        dataset: plan.dataset,
        model: plan.model,
        lossFunction: plan.lossFunction,
        lr: item.lr,
        epochs: plan.epochs,
        batchSz: plan.batchSz
      });
    }

    setBenchmarkStatus('Benchmark complete. Review Experiment History and Best Run.');
  } catch (err) {
    console.error('Benchmark error:', err);
    setBenchmarkStatus('Benchmark stopped: ' + err.message);
    alert('Benchmark failed: ' + err.message);
  } finally {
    benchmarkRunning = false;
    setButtonsRunning(false);
  }
}

async function runTrainingOnce({ optimizer, dataset, model, lossFunction, lr, epochs, batchSz }) {
  running = true;
  bestLoss = Infinity;
  lastRenderedEpoch = 0;
  localStartTime = Date.now();
  maxEpochs = epochs;

  resetTrainingUi(epochs);
  setStatus('Training');
  setButtonsRunning(true);
  resetLossChart(displayOptimizerName(optimizer));
  startArenaAnimation();
  setArenaProgress(0);
  startElapsedClock();

  const response = await fetch(`${API_BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      optimizer: String(optimizer).toLowerCase(),
      dataset,
      model,
      loss_function: lossFunction,
      learning_rate: lr,
      epochs,
      batch_size: batchSz
    })
  });

  const data = await response.json();
  if (!response.ok || data.status !== 'success') {
    throw new Error(data.message || 'Training failed');
  }

  while (true) {
    await sleep(900);
    const statusResponse = await fetch(`${API_BASE}/api/train/status`);
    const statusData = await statusResponse.json();
    if (!statusResponse.ok || statusData.status !== 'success') {
      throw new Error('Could not read training status');
    }

    const state = statusData.state || {};
    if (state.status === 'Training') {
      renderProgressState(state);
      continue;
    }

    if (state.status === 'Completed') {
      renderProgressState(state);
      if (state.result) applyBackendResult(state.result);
      finishTrainingUi('Completed');
      return state.result;
    }

    if (state.error) {
      throw new Error(state.error);
    }
  }
}

function benchmarkProtocol() {
  return {
    dataset: 'mnist',
    model: 'simplecnn',
    lossFunction: 'crossentropy',
    batchSz: 128,
    epochs: 2,
    optimizers: [
      { name: 'SGD', lr: 0.05 },
      { name: 'Momentum', lr: 0.05 },
      { name: 'RMSprop', lr: 0.001 },
      { name: 'Adam', lr: 0.001 },
      { name: 'AdamW', lr: 0.001 }
    ]
  };
}

function showBenchmarkPlan() {
  const panel = document.getElementById('benchmarkPlan');
  if (panel) panel.hidden = false;
  setText('benchmarkReason', 'Selected MNIST + Simple CNN + Cross-Entropy + batch size 128 + 2 epochs because it is fast, stable, and fair enough for comparing optimizer behavior in a live demo. Optimizers run sequentially instead of literally in parallel to avoid CPU/GPU overload and keep timing comparable.');
}

function setBenchmarkStatus(message) {
  setText('benchmarkStatus', message);
}

function setControlValue(id, value) {
  const select = document.getElementById(id);
  if (!select) return;
  select.value = value;
  const shell = select.nextElementSibling;
  const trigger = shell?.classList?.contains('custom-select')
    ? shell.querySelector('.custom-select-trigger')
    : null;
  if (trigger) {
    trigger.textContent = select.options[select.selectedIndex]?.textContent || value;
  }
  select.dispatchEvent(new Event('change', { bubbles: true }));
}

function sleep(ms) {
  return new Promise(resolve => window.setTimeout(resolve, ms));
}

async function pollTrainingStatus(isActiveRun) {
  if (!API_BASE) return;

  try {
    const response = await fetch(`${API_BASE}/api/train/status`);
    const data = await response.json();
    if (!response.ok || data.status !== 'success') return;

    const state = data.state || {};

    if (state.status === 'Training') {
      running = true;
      setStatus('Training');
      setButtonsRunning(true);
      renderProgressState(state);
      return;
    }

    if (state.status === 'Completed') {
      renderProgressState(state);
      if (state.result) applyBackendResult(state.result);
      finishTrainingUi('Completed');
      return;
    }

    if (state.error && isActiveRun) {
      alert('Training failed: ' + state.error);
      finishTrainingUi('Ready');
      return;
    }

    if (!isActiveRun) {
      setStatus('Ready');
      setButtonsRunning(false);
      stopElapsedClock();
    }
  } catch (err) {
    console.error('Status polling error:', err);
  }
}

function renderProgressState(state) {
  const metrics = state.metrics || {};
  const total = Number(state.total_epochs || maxEpochs || 0);
  const epoch = Number(state.current_epoch || 0);
  const progress = Number(state.progress || (total ? (epoch / total) * 100 : 0));

  if (total) {
    maxEpochs = total;
    setText('maxEpLabel', total);
  }

  setText('epochVal', epoch);
  setText('centerEpoch', `${epoch} / ${total || maxEpochs}`);
  setText('progLabel', `${epoch} / ${total || maxEpochs} epochs`);
  setProgress(progress);
  setArenaProgress(total ? epoch / total : 0);

  if (typeof state.elapsed_time_seconds === 'number') {
    setText('elapsedTime', formatSeconds(state.elapsed_time_seconds));
  }

  if (typeof metrics.train_loss === 'number') {
    bestLoss = Math.min(bestLoss, metrics.train_loss);
    setText('arenaLoss', metrics.train_loss.toFixed(4));
    setText('perfLoss', metrics.train_loss.toFixed(4));
    setText('perfBestLoss', bestLoss.toFixed(4));
  }

  if (typeof metrics.train_accuracy === 'number') {
    setText('arenaAcc', `${Math.round(metrics.train_accuracy)}%`);
    setText('perfAcc', `${metrics.train_accuracy.toFixed(2)}%`);
  }

  if (typeof metrics.validation_loss === 'number') {
    setText('perfValLoss', metrics.validation_loss.toFixed(4));
  }

  if (typeof metrics.validation_accuracy === 'number') {
    setText('perfValAcc', `${metrics.validation_accuracy.toFixed(2)}%`);
    setText('centerValAcc', `${metrics.validation_accuracy.toFixed(2)}%`);
  }

  if (typeof metrics.test_loss === 'number') {
    setText('perfTestLoss', metrics.test_loss.toFixed(4));
  }

  if (typeof metrics.test_accuracy === 'number') {
    setText('perfTestAcc', `${metrics.test_accuracy.toFixed(2)}%`);
    setText('centerTestAcc', `${metrics.test_accuracy.toFixed(2)}%`);
  }

  if (typeof metrics.epoch_time_seconds === 'number') {
    setText('perfTime', `${metrics.epoch_time_seconds.toFixed(2)} s`);
  }

  if (epoch > lastRenderedEpoch && typeof metrics.train_loss === 'number') {
    pushLossPoint(
      epoch,
      metrics.train_loss,
      Number(metrics.validation_loss ?? metrics.train_loss)
    );
    lastRenderedEpoch = epoch;
  }
}

function applyBackendResult(result) {
  const losses = result.train_losses || [];
  const valLosses = result.validation_losses || [];
  const trainAcc = Number(result.final_train_accuracy || 0);
  const valAcc = Number(result.final_validation_accuracy || 0);
  const testAcc = Number(result.final_test_accuracy || 0);
  const testLoss = Number(result.final_test_loss || 0);
  const finalTrainLoss = Number(result.final_train_loss || losses[losses.length - 1] || 0);
  const finalValLoss = Number(result.final_validation_loss || valLosses[valLosses.length - 1] || 0);

  scores[displayOptimizerName(result.optimizer)] = {
    acc: testAcc,
    loss: testLoss.toFixed(4),
    epochs: Number(result.epochs)
  };
  updateLeaderboard();

  setText('epochVal', result.epochs);
  setText('centerEpoch', `${result.epochs} / ${result.epochs}`);
  setText('progLabel', `${result.epochs} / ${result.epochs} epochs`);
  setProgress(100);
  setArenaProgress(1);

  setText('perfLoss', finalTrainLoss.toFixed(4));
  setText('perfBestLoss', Math.min(...losses.map(Number)).toFixed(4));
  setText('perfAcc', `${trainAcc.toFixed(2)}%`);
  setText('perfValLoss', finalValLoss.toFixed(4));
  setText('perfValAcc', `${valAcc.toFixed(2)}%`);
  setText('centerValAcc', `${valAcc.toFixed(2)}%`);
  setText('perfTestLoss', testLoss.toFixed(4));
  setText('perfTestAcc', `${testAcc.toFixed(2)}%`);
  setText('centerTestAcc', `${testAcc.toFixed(2)}%`);
  setText('perfTime', `${average(result.epoch_times || []).toFixed(2)} s`);
  setText('arenaLoss', testLoss.toFixed(4));
  setText('arenaAcc', `${testAcc.toFixed(2)}%`);
  setText('elapsedTime', formatSeconds(result.training_time_seconds));

  updateExperimentSummary(result);
  updateFinalResult(result);
  addRunHistory(result);
  renderTradeoffs(displayOptimizerName(result.optimizer), result);
}

function resetTrainingUi(epochs) {
  [
    'arenaLoss', 'arenaAcc', 'perfLoss', 'perfBestLoss', 'perfAcc',
    'perfValLoss', 'perfValAcc', 'perfTestLoss', 'perfTestAcc',
    'perfTime', 'expTotalTime', 'expTrainAcc', 'expValAcc', 'expLoss', 'expAcc',
    'centerValAcc', 'centerTestAcc'
  ].forEach(id => setText(id, '—'));

  setText('finalRing', '—');
  setText('finalHeadline', 'Experiment running');
  setText('finalSubtitle', 'The result story will update after the final test evaluation.');
  setText('finalNarrative', 'Training is collecting train and validation metrics before the final held-out test evaluation.');
  setText('maxEpLabel', epochs);
  setText('epochVal', '0');
  setText('centerEpoch', `0 / ${epochs}`);
  setText('progLabel', `0 / ${epochs} epochs`);
  setProgress(0);
  updateExperimentDraft();
}

function finishTrainingUi(status) {
  running = false;
  clearPollTimer();
  stopElapsedClock();
  stopArenaAnimation();
  setStatus(status);
  setButtonsRunning(false);
}

function setStatus(status) {
  const badge = document.getElementById('statusBadge');
  if (!badge) return;

  badge.textContent = status;
  setText('centerStatus', status);
  badge.className = 'header-badge';

  if (status === 'Training') badge.classList.add('running');
  if (status === 'Completed') badge.classList.add('done');
}

function setButtonsRunning(isRunning) {
  const runBtn = document.getElementById('runBtn');
  const startBtn = document.getElementById('startBtn');
  const benchmarkBtn = document.getElementById('benchmarkBtn');
  const disabled = isRunning || benchmarkRunning;

  if (runBtn) {
    runBtn.textContent = isRunning ? 'Training...' : '▶ Run Training';
    runBtn.classList.toggle('stopping', isRunning);
    runBtn.disabled = disabled;
  }

  if (startBtn) {
    startBtn.textContent = isRunning ? 'Training...' : '▶ Run Training';
    startBtn.disabled = disabled;
  }

  if (benchmarkBtn) {
    benchmarkBtn.textContent = benchmarkRunning ? 'Benchmark Running...' : 'Benchmark All Optimizers';
    benchmarkBtn.disabled = disabled;
  }
}

function startElapsedClock() {
  stopElapsedClock();
  elapsedTimer = window.setInterval(() => {
    if (!localStartTime) return;
    const elapsed = (Date.now() - localStartTime) / 1000;
    setText('elapsedTime', formatSeconds(elapsed));
  }, 250);
}

function stopElapsedClock() {
  if (elapsedTimer) window.clearInterval(elapsedTimer);
  elapsedTimer = null;
}

function clearPollTimer() {
  if (pollTimer) window.clearInterval(pollTimer);
  pollTimer = null;
}

// -- Summary Panels -------------------------------------------
function updateExperimentDraft() {
  const dataset = currentDatasetInfo();
  const model = currentModelInfo();
  const loss = currentLossInfo();

  setText('expOpt', window.activeOpt);
  setText('expDataset', dataset.label);
  setText('expModel', model.label);
  setText('expLossFn', loss.label);
  setText('expLR', document.getElementById('setLR')?.value || '—');
  setText('expBatch', document.getElementById('setBatch')?.value || '—');
  setText('expEpochs', document.getElementById('setEpochs')?.value || '—');
  updateLearningRateHint();
  updateBatchHint();
  updateEpochHint();
}

function updateExperimentSummary(result) {
  updateExperimentDraft();
  setText('expTotalTime', formatSeconds(result.training_time_seconds));
  setText('expTrainAcc', `${Number(result.final_train_accuracy).toFixed(2)}%`);
  setText('expValAcc', `${Number(result.final_validation_accuracy).toFixed(2)}%`);
  setText('expAcc', `${Number(result.final_test_accuracy).toFixed(2)}%`);
  setText('expLoss', Number(result.final_test_loss).toFixed(4));
}

function updateExperimentContext() {
  const dataset = currentDatasetInfo();
  const model = currentModelInfo();

  setText('datasetHelp', dataset.help);
  setText('modelHelp', model.help);
  setText('datasetTitle', `Dataset: ${dataset.label}`);
  setText('datasetDesc', dataset.desc + ' Each run uses 50,000 training samples, 10,000 validation samples, and 10,000 test samples.');
  setText('metaDataset', dataset.label);
  setText('metaModel', model.label);
  setText('protocolModel', model.label);
  updateLossContext();
  setText('researchCopy', `Run a controlled ${dataset.label} experiment with ${model.label} and compare convergence speed, validation behavior, and final test performance from the existing PyTorch backend.`);
  updateExperimentDraft();
}

function updateLossContext() {
  const loss = currentLossInfo();
  setText('lossHelp', loss.help);
  setText('metaLoss', loss.label);
  setText('protocolLoss', loss.label);
}

function updateLearningRateHint() {
  const lr = Number(document.getElementById('setLR')?.value || 0.001);
  setText('lrHelp', learningRateText(lr));
}

function learningRateText(lr) {
  let hint = 'Good default for Adam/AdamW and usually stable for CNN training.';
  if (lr < 0.001) hint = 'Very small step size: stable but may learn slowly.';
  if (lr >= 0.01 && lr < 0.1) hint = 'Moderate to large step size: useful for SGD/Momentum, sometimes sharp for adaptive optimizers.';
  if (lr >= 0.1) hint = 'Aggressive step size: included for trade-off demonstrations and may destabilize training.';
  return hint;
}

function updateBatchHint() {
  const batch = document.getElementById('setBatch')?.value || '64';
  setText('batchHelp', BATCH_INFO[batch] || BATCH_INFO[64]);
}

function updateEpochHint() {
  const epochs = document.getElementById('setEpochs')?.value || '2';
  setText('epochHelp', EPOCH_INFO[epochs] || EPOCH_INFO[2]);
}

function updateOptimizerLink(name) {
  const btn = document.getElementById('showCodeBtn');
  if (!btn) return;
  btn.href = OPT_LINKS[name] || OPT_LINKS.Adam;
  btn.textContent = `View ${name} implementation on GitHub`;
}

function updateFinalResult(result) {
  const optimizer = displayOptimizerName(result.optimizer);
  const dataset = DATASET_INFO[result.dataset]?.label || currentDatasetInfo().label;
  const model = MODEL_INFO[result.model]?.label || currentModelInfo().label;
  const loss = LOSS_CODE[result.loss_function]?.label || currentLossInfo().label;
  const valAcc = Number(result.final_validation_accuracy || 0);
  const testAcc = Number(result.final_test_accuracy || 0);
  const gap = Math.abs(valAcc - testAcc);
  const verdict = gap <= 2 ? 'strong generalization' : 'visible validation-test gap';

  setText('finalRing', `${testAcc.toFixed(1)}%`);
  setText('finalSubtitle', `${optimizer} on ${dataset} with ${model} and ${loss}`);
  setText('finalHeadline', `${optimizer} reached ${testAcc.toFixed(2)}% test accuracy`);
  setText(
    'finalNarrative',
    `Validation accuracy was ${valAcc.toFixed(2)}%, with a ${gap.toFixed(2)} point gap. That suggests ${verdict}. Total training time: ${formatSeconds(result.training_time_seconds)}.`
  );
}

function addRunHistory(result) {
  const row = {
    id: runHistory.length + 1,
    dataset: DATASET_INFO[result.dataset]?.label || currentDatasetInfo().label,
    model: MODEL_INFO[result.model]?.label || currentModelInfo().label,
    optimizer: displayOptimizerName(result.optimizer),
    loss: LOSS_CODE[result.loss_function]?.label || currentLossInfo().label,
    lr: result.learning_rate,
    testAcc: Number(result.final_test_accuracy || 0),
    valAcc: Number(result.final_validation_accuracy || 0),
    testLoss: Number(result.final_test_loss || 0),
    time: Number(result.training_time_seconds || 0)
  };

  runHistory.unshift(row);
  renderRunHistory();
}

function renderRunHistory() {
  const body = document.getElementById('runHistoryBody');
  if (!body) return;

  if (!runHistory.length) {
    body.innerHTML = '<tr><td colspan="8" class="history-empty">Run training to start comparing experiments.</td></tr>';
    updateBestRunCard(null);
    return;
  }

  const best = [...runHistory].sort((a, b) => b.testAcc - a.testAcc)[0];
  updateBestRunCard(best);

  body.innerHTML = runHistory.map(row => {
    const bestClass = row.id === best.id ? ' class="history-best-row"' : '';
    return `
      <tr${bestClass}>
        <td>${row.id}</td>
        <td>${row.dataset}</td>
        <td>${row.model}</td>
        <td>${row.optimizer}</td>
        <td>${row.loss}</td>
        <td>${row.lr}</td>
        <td class="acc-col">${row.testAcc.toFixed(2)}%</td>
        <td>${formatSeconds(row.time)}</td>
      </tr>
    `;
  }).join('');
}

function updateBestRunCard(best) {
  const card = document.getElementById('bestRunCard');
  if (!card) return;

  if (!best) {
    card.innerHTML = '<span>Best run</span><strong>Awaiting results</strong>';
    return;
  }

  card.innerHTML = `
    <span>Best run</span>
    <strong>${best.testAcc.toFixed(2)}%</strong>
    <em>${best.optimizer} · ${best.dataset}</em>
  `;
}

function exportRunHistoryCsv() {
  if (!runHistory.length) {
    alert('No completed runs to export yet.');
    return;
  }

  const headers = ['run', 'dataset', 'model', 'optimizer', 'loss', 'learning_rate', 'validation_accuracy', 'test_accuracy', 'test_loss', 'training_time_seconds'];
  const rows = runHistory
    .slice()
    .reverse()
    .map(row => [
      row.id,
      row.dataset,
      row.model,
      row.optimizer,
      row.loss,
      row.lr,
      row.valAcc.toFixed(2),
      row.testAcc.toFixed(2),
      row.testLoss.toFixed(4),
      row.time.toFixed(2)
    ]);

  const csv = [headers, ...rows]
    .map(items => items.map(csvCell).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `optimizer-quest-history-${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function resetRunHistory() {
  runHistory = [];
  renderRunHistory();
}

function csvCell(value) {
  const text = String(value ?? '');
  if (/[",\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function renderTradeoffs(name, result = null) {
  const info = OPT_INFO[name];
  const tradeoffItems = document.getElementById('tradeoffItems');
  if (!info || !tradeoffItems) return;

  const generalizationGap = result
    ? Math.abs(Number(result.final_validation_accuracy) - Number(result.final_test_accuracy)).toFixed(2)
    : null;
  const bestLossText = result
    ? `Best train loss reached ${Math.min(...result.train_losses.map(Number)).toFixed(4)} over ${result.epochs} epochs.`
    : `${name} is rated ${info.speed.toLowerCase()} for convergence in this comparison.`;
  const gapText = generalizationGap
    ? `Validation and test accuracy differ by ${generalizationGap} percentage points.`
    : 'Validation accuracy estimates model selection quality before final test reporting.';

  tradeoffItems.innerHTML = `
    <div class="tradeoff-item">
      <span class="tradeoff-icon">Speed</span>
      <div class="tradeoff-content">
        <span class="tradeoff-title">Convergence Pace</span>
        <span class="tradeoff-desc">${bestLossText}</span>
      </div>
    </div>
    <div class="tradeoff-item">
      <span class="tradeoff-icon">Gen</span>
      <div class="tradeoff-content">
        <span class="tradeoff-title">Generalization Check</span>
        <span class="tradeoff-desc">${gapText}</span>
      </div>
    </div>
    <div class="tradeoff-item">
      <span class="tradeoff-icon">LR</span>
      <div class="tradeoff-content">
        <span class="tradeoff-title">Hyperparameter Sensitivity</span>
        <span class="tradeoff-desc">${info.tips}</span>
      </div>
    </div>
  `;
}

// -- Leaderboard ----------------------------------------------
function updateLeaderboard() {
  const sorted = Object.entries(scores).sort((a, b) => b[1].acc - a[1].acc);
  const rankColors = ['#fbbf24', '#94a3b8', '#cd7c3a', '#4e6d96', '#334155'];

  const html = sorted.map(([name, s], i) => {
    const col = OPT_INFO[name]?.color || '#ffffff';
    return `
      <li class="lb-item">
        <span class="lb-rank" style="color:${rankColors[i] || '#334155'}">${i + 1}.</span>
        <span class="lb-name" style="color:${col}">${name}</span>
        <span class="lb-pts">${s.acc.toFixed(2)}% | ${s.loss}</span>
      </li>`;
  }).join('');

  const lb = document.getElementById('lbList');
  if (lb) {
    lb.innerHTML = html || '<li class="lb-empty">Run optimizers to compare</li>';
  }
}

// -- Small Helpers --------------------------------------------
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}

function setProgress(percent) {
  const progFill = document.getElementById('progFill');
  if (progFill) progFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
}

function formatSeconds(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((sum, item) => sum + Number(item || 0), 0) / values.length;
}

function currentDatasetInfo() {
  const key = document.getElementById('setDataset')?.value || 'mnist';
  return DATASET_INFO[key] || DATASET_INFO.mnist;
}

function currentModelInfo() {
  const key = document.getElementById('setModel')?.value || 'simplecnn';
  return MODEL_INFO[key] || MODEL_INFO.simplecnn;
}

function currentLossInfo() {
  const key = document.getElementById('setLossFn')?.value || 'crossentropy';
  return LOSS_CODE[key] || LOSS_CODE.crossentropy;
}

function displayOptimizerName(name) {
  const normalized = String(name || window.activeOpt).toLowerCase();
  const lookup = {
    sgd: 'SGD',
    momentum: 'Momentum',
    rmsprop: 'RMSprop',
    adam: 'Adam',
    adamw: 'AdamW'
  };
  return lookup[normalized] || window.activeOpt;
}
