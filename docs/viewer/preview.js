const params = new URLSearchParams(location.search);
const lab = params.get('lab');
const filename = params.get('file');
const status = document.getElementById('preview-status');
const preview = document.getElementById('preview');
const imageExtensions = new Set(['png', 'jpg', 'jpeg', 'webp']);
const videoExtensions = new Set(['mp4', 'webm']);

async function previewDocx(response) {
  const content = document.createElement('div');
  await window.docx.renderAsync(await response.blob(), content, null, {
    ignoreWidth: true,
    ignoreHeight: true,
    renderAltChunks: false,
    useBase64URL: true,
  });
  const frame = document.createElement('iframe');
  frame.title = 'Word document';
  frame.setAttribute('sandbox', '');
  frame.srcdoc = `<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; font-src data:; style-src 'unsafe-inline'">
    <title>Word document</title>
    <style>body{margin:0;overflow-wrap:anywhere} img{max-width:100%;height:auto}
    .docx-wrapper{padding:0!important} section.docx{padding:clamp(1rem,4vw,3rem)!important}</style>
    </head><body>${content.innerHTML}</body></html>`;
  return frame;
}

async function loadPreview() {
  const validFile = lab === null
    ? /^lab-0[1-7]\.(pdf|docx)$/.test(filename)
    : /^[1-7]$/.test(lab) && filename && !/[/\\]/.test(filename);
  if (!validFile) {
    status.textContent = 'Choose a file from the lab handouts.';
    return;
  }

  const extension = filename.split('.').pop().toLowerCase();
  const url = lab === null ? `../handouts/${filename}` : `/submissions/${lab}/${encodeURIComponent(filename)}`;
  document.getElementById('file-title').textContent = filename;
  document.title = `${filename} · CSST106`;
  if (lab !== null) {
    const back = document.getElementById('back');
    back.href = `/#lab-0${lab}`;
    back.textContent = `Lab 0${lab}`;
  }
  const download = document.getElementById('download');
  download.href = url;
  download.hidden = false;

  try {
    let element;
    if (extension === 'docx' || extension === 'py' || extension === 'ipynb') {
      const response = await fetch(url);
      if (!response.ok) throw new Error('File unavailable');
      if (extension === 'docx') {
        element = await previewDocx(response);
      } else {
        element = document.createElement('pre');
        element.textContent = await response.text();
      }
    } else {
      const inline = lab === null ? url : `${url}?preview=1`;
      const response = await fetch(inline, { method: 'HEAD' });
      if (!response.ok) throw new Error('File unavailable');
      if (extension === 'pdf') {
        element = document.createElement('iframe');
        element.title = 'PDF document';
      } else if (imageExtensions.has(extension)) {
        element = document.createElement('img');
        element.alt = filename;
      } else if (videoExtensions.has(extension)) {
        element = document.createElement('video');
        element.controls = true;
        element.playsInline = true;
        element.preload = 'metadata';
      } else {
        status.textContent = 'Download this file to open it.';
        return;
      }
      element.src = inline;
    }
    preview.append(element);
    status.hidden = true;
  } catch {
    status.textContent = 'Preview unavailable. Download the file to open it.';
  }
}

loadPreview();
