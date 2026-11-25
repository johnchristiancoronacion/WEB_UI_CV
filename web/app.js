const labs = [
  { title: 'Image Manipulation', nav: 'Image Manipulation', icon: 'image', description: 'Inspect pixel values and channels, then compare grayscale conversion, resizing, cropping, rotation, and flipping.' },
  { title: 'Edge Detection', nav: 'Edge Detection', icon: 'edge', description: 'Compare Sobel gradients and Canny edges, and see how smoothing and thresholds affect image boundaries.' },
  { title: 'Image Histograms', nav: 'Histograms', icon: 'chart', description: 'Measure the distribution of pixel intensities and compare global equalization, CLAHE, and contrast stretching.' },
  { title: 'SURF Feature Detection', nav: 'SURF Features', icon: 'feature', description: 'Detect and match local features across two views, then compare SURF with SIFT and ORB.' },
  { title: 'PCA and Eigenfaces', nav: 'PCA & Eigenfaces', icon: 'grid', description: 'Project face images into a smaller feature space and evaluate identity classification on held-out LFW portraits.' },
  { title: 'YOLO11 Object Detection', nav: 'YOLO11 Detection', icon: 'box', description: 'Detect and label objects in an image, video, or webcam feed using pretrained or custom YOLO11 weights.' },
  { title: 'Image Segmentation', nav: 'Segmentation', icon: 'layers', description: 'Compare thresholding, region growing, watershed, contour-based segmentation, and semantic scene labeling.' },
];

const $ = (id) => document.getElementById(id);
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const number = (value) => String(value).padStart(2, '0');
const inputVideo = $('source-video');
const inputImage = $('source-image');
const status = $('status');
const run = $('run');
let activeLab;
let media = null;
let comparison = null;
let previewUrl;
let liveStream;
let liveTimer;
let liveVersion = 0;
let activeRequest;
window.currentLabResult = null;

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function field(name, label, type, value, attributes = {}) {
  const wrapper = node('label', 'field');
  const text = node('span', '', label);
  const control = node(type === 'select' ? 'select' : 'input');
  control.name = name;
  control.id = `field-${name}`;
  if (type === 'select') {
    for (const [key, option] of Object.entries(attributes)) {
      const entry = node('option', '', option);
      entry.value = key;
      control.append(entry);
    }
    const selectWrap = node('span', 'select-wrap');
    selectWrap.append(control);
    selectWrap.insertAdjacentHTML('beforeend', icon('chevron'));
    wrapper.append(text, selectWrap);
  } else {
    control.type = type;
    for (const [key, attribute] of Object.entries(attributes)) control.setAttribute(key, attribute);
    wrapper.append(...(type === 'checkbox' ? [control, text] : [text, control]));
  }
  if (type === 'checkbox') control.checked = value;
  else control.value = value;
  return wrapper;
}

function num(name, label, value, min, max, step = 1) {
  return field(name, label, 'number', value, { min, max, step });
}

function fields(...items) {
  const group = node('div', 'fields');
  group.append(...items);
  return group;
}

function more(label, ...items) {
  const details = node('details', 'option-details');
  details.append(node('summary', '', label), ...items);
  return details;
}

function renderControls() {
  const controls = $('lab-controls');
  const form = $('lab-form');
  form.querySelector('#webcam')?.remove();
  $('training-results').hidden = activeLab !== 6;
  const sourceExtra = $('source-extra');
  sourceExtra.hidden = activeLab !== 4;
  sourceExtra.replaceChildren();
  controls.replaceChildren();

  if (activeLab === 1) {
    controls.append(fields(
      num('scale', 'Resize (%)', 65, 10, 200), num('rotation', 'Rotation (°)', 90, -180, 180),
      num('crop', 'Center crop (%)', 65, 10, 100), field('flip', 'Flip direction', 'select', 'horizontal', { horizontal: 'Horizontal', vertical: 'Vertical', both: 'Both axes' }),
    ), more('Pixel coordinates and custom crop', fields(
      num('pixel_x', 'Pixel X', '', 0, 8192), num('pixel_y', 'Pixel Y', '', 0, 8192),
      field('pixel_color', 'New pixel color', 'color', '#ff00ff'), node('div'),
      num('crop_x', 'Crop X', '', 0, 8192), num('crop_y', 'Crop Y', '', 0, 8192),
      num('crop_width', 'Crop width', '', 1, 8192), num('crop_height', 'Crop height', '', 1, 8192),
    ), node('p', 'help', 'Coordinates start at (0, 0) in the top-left corner. Fill all four crop fields to set a custom region.')));
  } else if (activeLab === 2) {
    controls.append(fields(num('low', 'Canny low threshold', 100, 0, 255), num('high', 'Canny high threshold', 200, 1, 255), num('blur', 'Gaussian kernel size', 3, 1, 15, 2)), node('p', 'help', 'Use an odd Gaussian kernel size. Higher thresholds keep stronger edges.'));
  } else if (activeLab === 3) {
    controls.append(fields(num('clip', 'CLAHE clip limit', 2, 0.5, 10, 0.5), num('tiles', 'CLAHE tiles per side', 8, 2, 16)), node('p', 'help', 'The built-in sample uses a compressed intensity range so the contrast adjustments are easy to compare.'));
  } else if (activeLab === 4) {
    controls.append(fields(num('threshold', 'Hessian threshold (%)', 1, 0.1, 20, 0.1), num('max_features', 'Features to display', 150, 10, 500), field('upright', 'Use upright SURF', 'checkbox', false)), node('p', 'help', 'Upload a second view to match features. With one image, the experiment compares it with a rotated copy.'));
    const compareInput = node('input', 'visually-hidden');
    compareInput.type = 'file';
    compareInput.accept = 'image/*';
    compareInput.tabIndex = -1;
    const compareButton = node('button', 'button button-light', 'Upload comparison image');
    const clear = node('button', 'button button-light', 'Clear comparison');
    clear.hidden = true;
    compareButton.addEventListener('click', () => compareInput.click());
    compareInput.addEventListener('change', () => {
      if (!compareInput.files[0]) return;
      comparison = compareInput.files[0];
      compareButton.textContent = comparison.name;
      clear.hidden = false;
      invalidate();
    });
    clear.addEventListener('click', () => {
      compareInput.value = '';
      comparison = null;
      clear.hidden = true;
      compareButton.textContent = 'Upload comparison image';
      invalidate();
    });
    sourceExtra.append(compareInput, compareButton, clear);
  } else if (activeLab === 5) {
    controls.append(fields(field('method', 'Experiment', 'select', 'eigenfaces', { eigenfaces: 'LFW face recognition', reconstruction: 'Image reconstruction' }), num('components', 'Principal components', 150, 1, 200)), node('p', 'help', 'The LFW experiment trains PCA and an SVM on a fixed training split, then scores held-out faces. Choose image reconstruction to use your own input.'));
    $('field-method').addEventListener('change', () => {
      const eigenfaces = $('field-method').value === 'eigenfaces';
      $('field-components').value = eigenfaces ? 150 : 20;
      selectSample();
      updateDatasetMode();
      invalidate();
    });
  } else if (activeLab === 6) {
    controls.append(fields(
      field('model', 'Model', 'select', 'pretrained', { pretrained: 'YOLO11n · COCO', custom: 'YOLO11n · Custom cards' }),
      num('confidence', 'Minimum confidence (%)', 25, 1, 99), num('iou', 'NMS overlap (%)', 45, 1, 99),
    ), node('p', 'help', 'Video runs sample up to four frames per second from the first 12 seconds. The webcam analyzes frames while the camera is active.'));
    const details = node('p', 'help');
    const training = node('a', '', 'Custom training and evaluation');
    training.href = '/report/index.html#lab-06';
    details.append(training);
    controls.append(details);
    $('field-model').addEventListener('change', () => {
      stopCamera();
      $('field-confidence').value = 25;
      if (!media) selectSample();
      invalidate();
    });
    const camera = node('button', 'button button-light', 'Start webcam');
    camera.type = 'button';
    camera.id = 'webcam';
    camera.addEventListener('click', () => liveStream ? stopCamera() : startCamera());
    form.querySelector('.form-actions').append(camera);
  } else {
    controls.append(fields(field('method', 'Segmentation method', 'select', 'otsu', {
      otsu: 'Otsu threshold', adaptive: 'Adaptive threshold', region: 'Seeded region growing',
      watershed: 'Watershed', edges: 'Edges and contours', semantic: 'SegFormer semantic', kmeans: 'K-means colors',
    })));
    const dynamic = node('div', 'method-settings');
    controls.append(dynamic);
    const updateMethod = () => {
      const method = $('field-method').value;
      const config = {
        otsu: [field('invert', 'Invert foreground', 'checkbox', false)],
        adaptive: [field('invert', 'Invert foreground', 'checkbox', false)],
        region: [num('seed_x', 'Seed X', '', 0, 8192), num('seed_y', 'Seed Y', '', 0, 8192), num('tolerance', 'Intensity tolerance', 20, 1, 100)],
        kmeans: [num('clusters', 'Color clusters', 3, 2, 8)],
      };
      dynamic.replaceChildren();
      if (config[method]) dynamic.append(fields(...config[method]));
      if (method === 'region') dynamic.append(node('p', 'help', 'Leave both coordinates blank to use the image center.'));
    };
    $('field-method').addEventListener('change', updateMethod);
    updateMethod();
  }
}

function setStatus(message, error = false) {
  status.textContent = message;
  status.toggleAttribute('data-error', error);
}

function clearResults() {
  window.currentLabResult = null;
  for (const id of ['metrics', 'video-result', 'outputs', 'detection-list']) {
    $(id).replaceChildren();
    $(id).hidden = id !== 'outputs';
  }
}

function invalidate(message = 'Run the experiment to see the results.') {
  activeRequest?.abort();
  activeRequest = null;
  run.disabled = Boolean(liveStream);
  clearResults();
  setStatus(message);
}

function updateDatasetMode() {
  const useDataset = activeLab === 5 && $('field-method').value === 'eigenfaces';
  $('dataset-preview').hidden = !useDataset;
  $('use-sample').hidden = useDataset;
  $('upload').hidden = useDataset;
  if (useDataset) {
    inputImage.hidden = true;
    inputVideo.hidden = true;
    $('source-name').textContent = 'Labeled Faces in the Wild';
    $('source-details').textContent = 'Built-in face recognition dataset';
    $('preview').classList.add('has-dataset');
  } else {
    $('preview').classList.remove('has-dataset');
  }
}

function updatePreview(file) {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : undefined;
  inputVideo.srcObject = null;
  inputVideo.removeAttribute('src');
  inputVideo.load();
  inputVideo.controls = true;
  inputVideo.muted = false;
  const isVideo = file && (file.type.startsWith('video/') || /\.(mp4|webm|avi|mov|mkv|m4v)$/i.test(file.name));
  const custom = activeLab === 6 && $('field-model').value === 'custom';
  inputImage.hidden = Boolean(isVideo);
  inputVideo.hidden = !isVideo;
  if (isVideo) inputVideo.src = previewUrl;
  else {
    inputImage.src = previewUrl || (custom ? '/custom-sample.jpg' : '/sample.jpg');
    inputImage.alt = file?.name || (custom ? 'Held-out custom card sample' : 'Street scene sample');
  }
  $('source-name').textContent = file?.name || (custom ? 'Held-out card example' : 'Street scene sample');
  $('source-details').textContent = file ? `${isVideo ? 'Uploaded video' : 'Uploaded image'} · ${(file.size / 1024).toFixed(0)} KB` : 'Sample image';
  updateDatasetMode();
}

function selectSample() {
  media = null;
  $('media').value = '';
  updatePreview(null);
}

function selectLab(lab) {
  if (lab === activeLab) return;
  stopCamera();
  invalidate();
  activeLab = lab;
  comparison = null;
  const selected = labs[lab - 1];
  $('current-lab').textContent = `Lab Activity ${number(lab)}`;
  $('lab-number').textContent = `Lab Activity ${number(lab)}`;
  $('lab-title').textContent = selected.title;
  $('lab-description').textContent = selected.description;
  document.title = `${selected.title} · CSST106 Vision Lab`;
  document.querySelector('.hero-pattern use').setAttribute('href', `#i-${selected.icon}`);
  for (const link of $('lab-navigation').children) {
    if (Number(link.dataset.lab) === lab) link.setAttribute('aria-current', 'page');
    else link.removeAttribute('aria-current');
  }
  $('media').accept = lab === 6 ? 'image/*,video/*,.avi,.mkv' : 'image/*';
  $('upload-label').textContent = lab === 6 ? 'Upload media' : 'Upload image';
  renderControls();
  selectSample();
  window.scrollTo(0, 0);
  execute();
}

function openImage(output) {
  $('dialog-title').textContent = output.label;
  $('dialog-image').src = output.source;
  $('dialog-image').alt = output.label;
  const download = $('download');
  download.href = output.source;
  download.download = `lab-${number(activeLab)}-${output.label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/-$/, '')}.png`;
  $('image-dialog').showModal();
}

function showResults(result, live = false) {
  window.currentLabResult = result;
  if (result.input && !live && !inputImage.hidden) {
    inputImage.src = result.input.source;
    $('source-details').textContent = `${result.input.width} × ${result.input.height} px`;
  }
  const metrics = $('metrics');
  metrics.replaceChildren(...Object.entries(result.metrics).map(([label, value]) => {
    const metric = node('div', 'metric', label);
    metric.append(node('strong', '', Array.isArray(value) ? value.join(', ') : value));
    return metric;
  }));
  metrics.hidden = metrics.childElementCount === 0;

  const outputs = $('outputs');
  outputs.replaceChildren(...result.outputs.map((output, index) => {
    const button = node('button', 'output-card');
    button.classList.toggle('wide', result.outputs.length === 1 || (activeLab === 2 && index === 0) || (activeLab === 3 && index === 0));
    const label = node('span', 'output-label', output.label);
    label.insertAdjacentHTML('beforeend', icon('expand'));
    const image = node('img');
    image.alt = '';
    image.src = output.source;
    const preview = node('span', 'output-media');
    preview.append(image);
    button.append(preview, label);
    button.addEventListener('click', () => openImage(output));
    return button;
  }));

  const videoResult = $('video-result');
  if (result.video) {
    const player = node('video');
    player.controls = true;
    player.playsInline = true;
    player.src = result.video;
    const link = node('a', 'button button-light', 'Download annotated video');
    link.href = result.video;
    link.download = 'lab-06-detections.webm';
    videoResult.replaceChildren(player, link);
  } else videoResult.replaceChildren();
  videoResult.hidden = !result.video;

  const detectionList = $('detection-list');
  detectionList.replaceChildren();
  detectionList.hidden = !result.detections?.length;
  if (result.detections?.length) {
    const list = node('ul');
    for (const item of result.detections) list.append(node('li', '', `${item.class_name} · ${(item.confidence * 100).toFixed(1)}%`));
    detectionList.append(node('h3', '', 'Detected objects'), list);
  }
  setStatus(live ? 'Webcam detection is running.' : `${result.outputs.length} ${result.outputs.length === 1 ? 'output' : 'outputs'} ready`);
}

async function execute(frame = null) {
  const controller = new AbortController();
  activeRequest?.abort();
  activeRequest = controller;
  const lab = activeLab;
  const data = new FormData($('lab-form'));
  for (const [key, value] of [...data.entries()]) if (value === '') data.delete(key);
  if (frame) {
    data.set('media', frame, 'webcam.jpg');
    data.set('live', '1');
  } else if (media) data.set('media', media);
  if (comparison) data.set('comparison', comparison);
  if (!frame) {
    clearResults();
    run.disabled = true;
    setStatus('Processing…');
  }
  try {
    const response = await fetch(`/api/labs/${lab}`, { method: 'POST', body: data, signal: controller.signal });
    const result = await response.json();
    if (controller.signal.aborted) return false;
    if (!response.ok) throw new Error(result.error || 'The experiment could not finish.');
    showResults(result, Boolean(frame));
    return true;
  } catch (error) {
    if (!controller.signal.aborted) setStatus(error.message, true);
    return false;
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      run.disabled = Boolean(liveStream);
    }
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Open the app on localhost or HTTPS to use the webcam.', true);
    return;
  }
  try {
    const version = ++liveVersion;
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 } }, audio: false });
    if (activeLab !== 6 || version !== liveVersion) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    invalidate('Starting webcam detection…');
    liveStream = stream;
    run.disabled = true;
    $('webcam').textContent = 'Stop webcam';
    $('source-name').textContent = 'Webcam';
    $('source-details').textContent = 'Live input';
    inputImage.hidden = true;
    inputVideo.hidden = false;
    inputVideo.controls = false;
    inputVideo.muted = true;
    inputVideo.srcObject = stream;
    await inputVideo.play();
    const canvas = node('canvas');
    const context = canvas.getContext('2d');
    const tick = async () => {
      if (stream !== liveStream) return;
      const scale = Math.min(1, 640 / inputVideo.videoWidth);
      canvas.width = Math.round(inputVideo.videoWidth * scale);
      canvas.height = Math.round(inputVideo.videoHeight * scale);
      context.drawImage(inputVideo, 0, 0, canvas.width, canvas.height);
      const frame = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .8));
      if (!frame || stream !== liveStream) return;
      if (await execute(frame)) liveTimer = setTimeout(tick, 250);
      else if (stream === liveStream) stopCamera(false);
    };
    tick();
  } catch {
    stopCamera(false);
    setStatus('The webcam is unavailable. Check camera access or upload a file.', true);
  }
}

function stopCamera(refresh = true) {
  liveVersion += 1;
  if (!liveStream) return;
  liveStream.getTracks().forEach((track) => track.stop());
  liveStream = undefined;
  clearTimeout(liveTimer);
  activeRequest?.abort();
  run.disabled = false;
  $('webcam').textContent = 'Start webcam';
  updatePreview(media);
  if (refresh) setStatus(window.currentLabResult ? 'Webcam stopped. The last annotated frame is shown.' : 'Webcam stopped.');
}

async function showTraining() {
  const details = $('training-results');
  const content = $('training-content');
  if (!details.open || content.dataset.loaded) return;
  content.textContent = 'Loading training results…';
  try {
    const response = await fetch('/api/lab-06/training');
    if (!response.ok) throw new Error('Training results are unavailable.');
    const result = await response.json();
    const summary = node('p', '', `${result.dataset}: ${result.classes.join(', ')}. ${result.splits.train} training, ${result.splits.val} validation, and ${result.splits.test} test images. ${result.epochs} epochs at ${result.image_size} × ${result.image_size} px.`);
    const annotation = node('p', '', 'YOLO annotations record the class ID and normalized box center, width, and height for each object.');
    const table = node('table');
    const headings = node('tr');
    for (const name of ['Dataset split', 'Precision', 'Recall', 'mAP50', 'mAP50–95']) {
      const cell = node('th', '', name);
      cell.scope = 'col';
      headings.append(cell);
    }
    table.createTHead().append(headings);
    const body = table.createTBody();
    for (const [label, split] of [['Validation', result.validation], ['Held-out test', result.test]]) {
      const row = body.insertRow();
      const title = node('th', '', label);
      title.scope = 'row';
      row.append(title);
      for (const key of ['precision', 'recall', 'map50', 'map50_95']) row.insertCell().textContent = `${(split[key] * 100).toFixed(2)}%`;
    }
    const tableWrap = node('div', 'table-wrap');
    tableWrap.append(table);
    content.replaceChildren(summary, annotation, tableWrap);
    content.dataset.loaded = 'true';
  } catch (error) {
    content.textContent = error.message;
  }
}

function openCurrentLab() {
  const lab = Number(location.hash.match(/^#lab-0?([1-7])$/)?.[1]) || 1;
  selectLab(lab);
  if (matchMedia('(max-width: 820px)').matches) {
    const link = $('lab-navigation').querySelector(`[data-lab="${lab}"]`);
    $('lab-navigation').scrollLeft = Math.max(0, link.offsetLeft - 15);
  }
}

for (const [index, lab] of labs.entries()) {
  const link = node('a', 'lab-link');
  link.href = `#lab-${number(index + 1)}`;
  link.dataset.lab = index + 1;
  link.insertAdjacentHTML('beforeend', icon(lab.icon));
  const label = node('span');
  label.append(node('small', '', `Lab Activity ${number(index + 1)}`), document.createTextNode(lab.nav));
  link.append(label);
  $('lab-navigation').append(link);
}
$('lab-form').addEventListener('submit', (event) => {
  event.preventDefault();
  execute();
});
$('upload').addEventListener('click', () => $('media').click());
$('media').addEventListener('change', () => {
  if (!$('media').files[0]) return;
  stopCamera();
  media = $('media').files[0];
  updatePreview(media);
  invalidate();
});
$('use-sample').addEventListener('click', () => {
  stopCamera();
  selectSample();
  execute();
});
window.addEventListener('hashchange', openCurrentLab);
window.addEventListener('pagehide', () => stopCamera(false));
$('training-results').addEventListener('toggle', showTraining);
openCurrentLab();
