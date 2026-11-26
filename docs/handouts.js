async function loadHandout(lab) {
  const status = lab.querySelector('.status');
  const title = lab.querySelector('h2').textContent;

  try {
    for (const extension of ['pdf', 'docx']) {
      const filename = `${lab.id}.${extension}`;
      const path = `handouts/${filename}`;
      const response = await fetch(path, { method: 'HEAD' });
      if (response.status === 404) continue;
      if (!response.ok) throw new Error('Handout check failed.');

      const view = lab.querySelector('.handout-link');
      view.href = `viewer/?${new URLSearchParams({ file: filename })}`;
      view.setAttribute('aria-label', `View handout for ${title}`);
      const download = lab.querySelector('.download-link');
      download.href = path;
      download.setAttribute('aria-label', `Download handout for ${title}`);
      status.hidden = true;
      lab.querySelector('.handout-actions').hidden = false;
      return;
    }
    status.textContent = 'Handout pending';
  } catch {
    status.textContent = 'Could not check handout';
  }
}

document.querySelectorAll('.labs > li').forEach(loadHandout);
