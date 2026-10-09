// Set di icone (24×24, tratto 1,5 px, angoli arrotondati) disegnate per Sirio OCR.

import { Raw } from './util.js';

const P = {
  dashboard: '<rect x="3.5" y="3.5" width="7" height="8" rx="1.6"/><rect x="13.5" y="3.5" width="7" height="5" rx="1.6"/><rect x="13.5" y="11.5" width="7" height="9" rx="1.6"/><rect x="3.5" y="14.5" width="7" height="6" rx="1.6"/>',
  documents: '<path d="M8 3.5h6.5L19 8v10.5a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2z"/><path d="M14 3.5V8h5"/><path d="M9.5 12.5h6M9.5 16h4"/>',
  file: '<path d="M7 3.5h7L18.5 8v10.5a2 2 0 0 1-2 2h-9.5a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2z"/><path d="M13.5 3.5V8h5"/>',
  sheet: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10"/>',
  excel: '<rect x="3.5" y="4.5" width="17" height="15" rx="2.2"/><path d="m8.5 9 4.5 6M13 9l-4.5 6"/><path d="M15.5 9h2M15.5 12h2M15.5 15h2"/>',
  settings: '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  upload: '<path d="M12 15.5V4.5M7.5 9 12 4.5 16.5 9"/><path d="M4.5 15v2.5a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V15"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  'chevron-left': '<path d="m14.5 6-6 6 6 6"/>',
  'chevron-right': '<path d="m9.5 6 6 6-6 6"/>',
  'chevron-down': '<path d="m6 9.5 6 6 6-6"/>',
  'chevron-up': '<path d="m6 14.5 6-6 6 6"/>',
  'arrow-left': '<path d="M19 12H5M11 6l-6 6 6 6"/>',
  'arrow-right': '<path d="M5 12h14M13 6l6 6-6 6"/>',
  'arrow-up': '<path d="M12 19V5M6 11l6-6 6 6"/>',
  'arrow-down': '<path d="M12 5v14M6 13l6 6 6-6"/>',
  'alert-triangle': '<path d="M10.3 4.2 2.9 17.1A2 2 0 0 0 4.6 20h14.8a2 2 0 0 0 1.7-2.9L13.7 4.2a2 2 0 0 0-3.4 0z"/><path d="M12 9.5v4M12 16.8v.2"/>',
  'alert-circle': '<circle cx="12" cy="12" r="8.5"/><path d="M12 8v4.5M12 15.8v.2"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5M12 8v.2"/>',
  'check-circle': '<circle cx="12" cy="12" r="8.5"/><path d="m8.5 12.2 2.4 2.4 4.6-4.8"/>',
  'x-circle': '<circle cx="12" cy="12" r="8.5"/><path d="m9.2 9.2 5.6 5.6M14.8 9.2l-5.6 5.6"/>',
  slash: '<circle cx="12" cy="12" r="8.5"/><path d="m6 6 12 12"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
  'eye-off': '<path d="M3 3l18 18"/><path d="M10.6 5.6A9.7 9.7 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-2.6 3.4M6.6 6.6C4 8.3 2.5 12 2.5 12S6 18.5 12 18.5c1.6 0 3-.4 4.3-1.1"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/>',
  key: '<circle cx="8" cy="15" r="3.5"/><path d="m10.5 12.5 8-8M16 7l2.5 2.5M14 9l2 2"/>',
  cpu: '<rect x="6.5" y="6.5" width="11" height="11" rx="2"/><rect x="9.5" y="9.5" width="5" height="5" rx="1"/><path d="M9.5 3.5v3M14.5 3.5v3M9.5 17.5v3M14.5 17.5v3M3.5 9.5h3M3.5 14.5h3M17.5 9.5h3M17.5 14.5h3"/>',
  sparkle: '<path d="M12 3.5c.6 4.4 2.9 6.9 8.5 8.5-5.6 1.6-7.9 4.1-8.5 8.5-.6-4.4-2.9-6.9-8.5-8.5 5.6-1.6 7.9-4.1 8.5-8.5z"/>',
  cloud: '<path d="M7 18.5a4.5 4.5 0 0 1-.6-9 6 6 0 0 1 11.4 1.6A3.8 3.8 0 0 1 17.5 18.5z"/>',
  folder: '<path d="M3.5 7.5a2 2 0 0 1 2-2h3.6l2 2h7.4a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  'folder-open': '<path d="M3.5 17.5v-10a2 2 0 0 1 2-2h3.6l2 2h6.4a2 2 0 0 1 2 2v1"/><path d="M3.5 17.5 6 11.6a1.5 1.5 0 0 1 1.4-1h12.3a1 1 0 0 1 .9 1.4l-2.3 5.5a1.5 1.5 0 0 1-1.4 1H5a1.5 1.5 0 0 1-1.5-1z"/>',
  download: '<path d="M12 4.5v11M7.5 11 12 15.5 16.5 11"/><path d="M4.5 18.5h15"/>',
  external: '<path d="M13.5 4.5h6v6M19.5 4.5l-8 8"/><path d="M18 14v3.5a2 2 0 0 1-2 2H6.5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2H10"/>',
  refresh: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.5 4.5v4h-4"/>',
  trash: '<path d="M4.5 7h15M10 4.5h4M6.5 7l.8 11.2a2 2 0 0 0 2 1.8h5.4a2 2 0 0 0 2-1.8L17.5 7"/><path d="M10 11v5M14 11v5"/>',
  'zoom-in': '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2M8.5 11h5M11 8.5v5"/>',
  'zoom-out': '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2M8.5 11h5"/>',
  fit: '<path d="M4.5 9V5.5a1 1 0 0 1 1-1H9M15 4.5h3.5a1 1 0 0 1 1 1V9M19.5 15v3.5a1 1 0 0 1-1 1H15M9 19.5H5.5a1 1 0 0 1-1-1V15"/>',
  'fit-width': '<path d="M4.5 5v14M19.5 5v14M8 12h8M10.5 9.5 8 12l2.5 2.5M13.5 9.5 16 12l-2.5 2.5"/>',
  crosshair: '<circle cx="12" cy="12" r="7.5"/><path d="M12 2.5v4M12 17.5v4M2.5 12h4M17.5 12h4"/><circle cx="12" cy="12" r="1.2"/>',
  layers: '<path d="m12 3.5 8.5 4.5L12 12.5 3.5 8z"/><path d="m3.5 12 8.5 4.5 8.5-4.5"/><path d="m3.5 16 8.5 4.5 8.5-4.5"/>',
  sun: '<circle cx="12" cy="12" r="3.8"/><path d="M12 3v1.8M12 19.2V21M3 12h1.8M19.2 12H21M5.6 5.6l1.3 1.3M17.1 17.1l1.3 1.3M5.6 18.4l1.3-1.3M17.1 6.9l1.3-1.3"/>',
  moon: '<path d="M19.5 14.5A8 8 0 0 1 9.5 4.5a8 8 0 1 0 10 10z"/>',
  monitor: '<rect x="3.5" y="4.5" width="17" height="12" rx="2"/><path d="M8.5 20h7M12 16.5V20"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  edit: '<path d="M14.5 5.5l4 4"/><path d="M4.5 19.5l1-4.2L15.6 5.2a1.6 1.6 0 0 1 2.3 0l.9.9a1.6 1.6 0 0 1 0 2.3L8.7 18.5z"/>',
  signature: '<path d="M3.5 16.5c2.5 0 4-6.5 5.5-6.5s-1 6.5 1 6.5 2.5-3.5 4-3.5 1 3.5 2.5 3.5 2-1.5 4-1.5"/><path d="M3.5 20h17"/>',
  sidebar: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M9.5 4.5v15"/>',
  keyboard: '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6 9.5h.01M9 9.5h.01M12 9.5h.01M15 9.5h.01M18 9.5h.01M6 12.5h.01M9 12.5h.01M12 12.5h.01M15 12.5h.01M18 12.5h.01M8 15.5h8"/>',
  filter: '<path d="M4 5.5h16l-6.2 7.3v5.4l-3.6 1.8v-7.2z"/>',
  sort: '<path d="M8 5v14M4.5 15.5 8 19l3.5-3.5M16 19V5M12.5 8.5 16 5l3.5 3.5"/>',
  more: '<circle cx="6" cy="12" r="1.1"/><circle cx="12" cy="12" r="1.1"/><circle cx="18" cy="12" r="1.1"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2"/><path d="M15.5 8.5V6a1.5 1.5 0 0 0-1.5-1.5H6A1.5 1.5 0 0 0 4.5 6v8A1.5 1.5 0 0 0 6 15.5h2.5"/>',
  undo: '<path d="M9 14.5 4.5 10 9 5.5"/><path d="M4.5 10h10a5 5 0 0 1 0 10H11"/>',
  redo: '<path d="m15 14.5 4.5-4.5L15 5.5"/><path d="M19.5 10h-10a5 5 0 0 0 0 10H13"/>',
  save: '<path d="M5.5 3.5h10l3 3v12a2 2 0 0 1-2 2h-11a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2z"/><path d="M8 3.5v4h7v-4M7.5 20.5v-6h9v6"/>',
  shield: '<path d="M12 3.5 5 6v5.5c0 4.3 3 7.6 7 9 4-1.4 7-4.7 7-9V6z"/><path d="m9 12 2.2 2.2L15.5 10"/>',
  lock: '<rect x="5" y="10.5" width="14" height="10" rx="2"/><path d="M8 10.5V8a4 4 0 0 1 8 0v2.5"/>',
  list: '<path d="M9 6.5h11M9 12h11M9 17.5h11M4.5 6.5h.01M4.5 12h.01M4.5 17.5h.01"/>',
  cards: '<rect x="3.5" y="3.5" width="7" height="7" rx="1.6"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.6"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.6"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.6"/>',
  image: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="9.5" r="1.8"/><path d="m20.5 15.5-4.5-4.5-9 8.5"/>',
  flag: '<path d="M5.5 20.5v-16"/><path d="M5.5 4.5h11.5l-2.2 4 2.2 4H5.5"/>',
  user: '<circle cx="12" cy="8.5" r="3.5"/><path d="M5 19.5c1.2-3.3 3.8-5 7-5s5.8 1.7 7 5"/>',
  school: '<path d="M3.5 9.5 12 5l8.5 4.5"/><path d="M5.5 10.5v8M18.5 10.5v8M9.5 18.5v-5h5v5M3.5 18.5h17"/>',
  calendar: '<rect x="3.5" y="5" width="17" height="15" rx="2"/><path d="M3.5 9.5h17M8 3v4M16 3v4"/>',
  coin: '<circle cx="12" cy="12" r="8.5"/><path d="M14.6 9.3c-.4-.9-1.4-1.5-2.6-1.5-1.5 0-2.6.8-2.6 2s1.1 1.7 2.6 2 2.6.8 2.6 2-1.1 2-2.6 2c-1.2 0-2.2-.6-2.6-1.5M12 6.3v1.5M12 16.2v1.5"/>',
  scan: '<path d="M4.5 8V5.5a1 1 0 0 1 1-1H8M16 4.5h2.5a1 1 0 0 1 1 1V8M19.5 16v2.5a1 1 0 0 1-1 1H16M8 19.5H5.5a1 1 0 0 1-1-1V16M7.5 12h9"/>',
  help: '<circle cx="12" cy="12" r="8.5"/><path d="M9.6 9.5a2.5 2.5 0 0 1 4.8.8c0 1.7-2.4 2.2-2.4 3.7M12 16.8v.2"/>',
  palette: '<path d="M12 3.5a8.5 8.5 0 1 0 0 17c1.1 0 1.6-.8 1.6-1.6 0-1.2-.9-1.4-.9-2.4 0-.9.7-1.5 1.6-1.5h2a3.7 3.7 0 0 0 3.7-3.7C20 7 16.4 3.5 12 3.5z"/><circle cx="7.8" cy="11" r="1"/><circle cx="10.5" cy="7.5" r="1"/><circle cx="15" cy="8" r="1"/>',
  gauge: '<path d="M4.5 16.5a8 8 0 1 1 15 0"/><path d="m12 13.5 3.5-4"/><circle cx="12" cy="14" r="1.3"/>',
  layers2: '<rect x="4" y="4" width="11" height="11" rx="2"/><path d="M9 19h9a2 2 0 0 0 2-2V8"/>',
  hourglass: '<path d="M6.5 3.5h11M6.5 20.5h11M7.5 3.5c0 4.5 4.5 5.5 4.5 8.5s-4.5 4-4.5 8.5M16.5 3.5c0 4.5-4.5 5.5-4.5 8.5s4.5 4 4.5 8.5"/>',
  dash: '<path d="M7 12h10"/>',
  play: '<path d="M8 5.5v13l10-6.5z"/>',
  power: '<path d="M12 3.5v8"/><path d="M7 6.5a7.5 7.5 0 1 0 10 0"/>',
  wifi: '<path d="M2.5 9a14 14 0 0 1 19 0M5.5 12.5a9.5 9.5 0 0 1 13 0M8.5 16a5 5 0 0 1 7 0"/><path d="M12 19.5v.01"/>',
  pen: '<path d="M4.5 19.5h4l10-10a2.8 2.8 0 0 0-4-4l-10 10z"/><path d="m13.5 6.5 4 4"/>',
  stamp: '<path d="M9.5 13.5V11a2.5 2.5 0 1 1 5 0v2.5"/><path d="M5 13.5h14l.5 3h-15z"/><path d="M6 20h12"/>',
};

/**
 * SVG dell'icona come HTML attendibile (Raw): non viene sottoposto a escape nei template
 * `html` e si converte in stringa negli altri contesti.
 */
export function icon(name, cls = '') {
  const body = P[name];
  if (!body) return new Raw('');
  return new Raw(`<svg class="icon${cls ? ' ' + cls : ''}" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${body}</svg>`);
}

export const CHECK_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>';
export const CROSS_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/></svg>';

/* Illustrazioni per gli stati vuoti (colori dai token CSS). */
export const ART = {
  upload: `<svg class="dropzone-art" viewBox="0 0 170 120" aria-hidden="true">
    <defs><linearGradient id="dz-g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#60A5FA"/><stop offset="1" stop-color="#2563EB"/></linearGradient></defs>
    <ellipse cx="85" cy="110" rx="58" ry="6" fill="currentColor" opacity=".06"/>
    <g transform="rotate(-9 52 62)"><rect class="art-paper-2" x="30" y="22" width="56" height="74" rx="7" stroke-width="1.5"/>
      <path class="art-line" d="M40 40h30M40 50h36M40 60h24" stroke-width="2"/></g>
    <g transform="rotate(8 118 62)"><rect class="art-paper-2" x="88" y="22" width="56" height="74" rx="7" stroke-width="1.5"/>
      <path class="art-line" d="M98 40h30M98 50h36M98 60h22" stroke-width="2"/></g>
    <rect class="art-paper" x="55" y="12" width="62" height="82" rx="8" stroke-width="1.5"/>
    <path class="art-line" d="M66 30h28M66 40h40M66 50h40M66 60h40M66 70h40" stroke-width="1.5" opacity=".7"/>
    <path class="art-ink" d="M67 79c4-6 7 4 10-1s5-4 8 0 6-3 9 1" stroke-width="2"/>
    <circle cx="117" cy="88" r="17" fill="url(#dz-g)"/>
    <path d="M117 96V80M110.5 86.5 117 80l6.5 6.5" stroke="#fff" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M42 14c.5 3.5 2 5 5.5 5.5-3.5.5-5 2-5.5 5.5-.5-3.5-2-5-5.5-5.5 3.5-.5 5-2 5.5-5.5z" class="art-accent" opacity=".85"/>
    <path d="M140 18c.3 2.3 1.3 3.3 3.6 3.6-2.3.3-3.3 1.3-3.6 3.6-.3-2.3-1.3-3.3-3.6-3.6 2.3-.3 3.3-1.3 3.6-3.6z" class="art-accent" opacity=".5"/>
  </svg>`,
  documents: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <ellipse cx="74" cy="104" rx="50" ry="5" fill="currentColor" opacity=".06"/>
    <rect class="art-soft" x="20" y="16" width="108" height="80" rx="14"/>
    <rect class="art-paper" x="44" y="10" width="60" height="80" rx="8" stroke-width="1.5"/>
    <path class="art-line" d="M54 28h30M54 38h40M54 48h40M54 58h26" stroke-width="1.8"/>
    <path class="art-ink" d="M55 74c4-6 7 4 10-1s5-4 8 0 6-3 9 1" stroke-width="2"/>
    <path d="M118 22c.6 4.2 2.4 6 6.6 6.6-4.2.6-6 2.4-6.6 6.6-.6-4.2-2.4-6-6.6-6.6 4.2-.6 6-2.4 6.6-6.6z" class="art-accent"/>
  </svg>`,
  queue: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <ellipse cx="74" cy="104" rx="46" ry="5" fill="currentColor" opacity=".06"/>
    <rect class="art-paper" x="30" y="22" width="88" height="20" rx="7" stroke-width="1.5"/>
    <rect class="art-paper" x="30" y="48" width="88" height="20" rx="7" stroke-width="1.5"/>
    <rect class="art-paper" x="30" y="74" width="88" height="20" rx="7" stroke-width="1.5"/>
    <rect class="art-soft" x="38" y="29" width="50" height="6" rx="3"/><rect class="art-accent" x="38" y="29" width="50" height="6" rx="3" opacity=".9"/>
    <rect class="art-soft" x="38" y="55" width="60" height="6" rx="3"/><rect class="art-accent" x="38" y="55" width="38" height="6" rx="3" opacity=".7"/>
    <rect class="art-soft" x="38" y="81" width="60" height="6" rx="3"/>
    <circle cx="106" cy="32" r="5" class="art-ok"/><path d="m103.6 32 1.7 1.7 3-3.2" stroke="#fff" stroke-width="1.6" fill="none" stroke-linecap="round"/>
  </svg>`,
  sheet: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <ellipse cx="74" cy="104" rx="50" ry="5" fill="currentColor" opacity=".06"/>
    <rect class="art-paper" x="26" y="14" width="96" height="80" rx="10" stroke-width="1.5"/>
    <path class="art-line" d="M26 34h96M26 50h96M26 66h96M26 82h96M52 14v80M78 14v80M100 14v80" stroke-width="1.2" opacity=".8"/>
    <rect x="26.75" y="14.75" width="94.5" height="19" rx="9" fill="#107C41" opacity=".14"/>
    <rect x="53" y="51" width="24" height="14" fill="#3B82F6" opacity=".18"/><rect x="53" y="51" width="24" height="14" fill="none" stroke="#3B82F6" stroke-width="1.6"/>
  </svg>`,
  allgood: `<svg class="empty-art" viewBox="0 0 148 112" aria-hidden="true">
    <ellipse cx="74" cy="104" rx="40" ry="5" fill="currentColor" opacity=".06"/>
    <circle cx="74" cy="54" r="36" class="art-soft"/>
    <circle cx="74" cy="54" r="24" class="art-ok"/>
    <path d="m63 54 7.5 7.5L86 46" stroke="#fff" stroke-width="3.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M114 22c.6 4.2 2.4 6 6.6 6.6-4.2.6-6 2.4-6.6 6.6-.6-4.2-2.4-6-6.6-6.6 4.2-.6 6-2.4 6.6-6.6z" class="art-accent" opacity=".8"/>
  </svg>`,
};
