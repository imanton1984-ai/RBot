export function formatPrice(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '-';

  const v = Number(value);
  if (v === 0) return '0';

  const abs = Math.abs(v);

  // Dynamic precision based on price magnitude
  // For very small prices (e.g., 0.00876), show enough significant digits
  let decimals: number;
  if (abs >= 1000) decimals = 2;
  else if (abs >= 100) decimals = 3;
  else if (abs >= 10) decimals = 4;
  else if (abs >= 1) decimals = 5;
  else if (abs >= 0.1) decimals = 6;
  else if (abs >= 0.01) decimals = 7;
  else if (abs >= 0.001) decimals = 8;
  else if (abs >= 0.0001) decimals = 9;
  else decimals = 10;

  // Format with fixed decimals, then remove only unnecessary trailing zeros
  // but keep at least 2 decimal places for readability
  const formatted = v.toFixed(decimals);

  // Remove trailing zeros but keep minimum meaningful precision
  // e.g., "0.00876000" → "0.00876", "45000.00" → "45000"
  const parts = formatted.split('.');
  if (parts.length === 1) return parts[0];

  // Trim trailing zeros from decimal part
  let dec = parts[1].replace(/0+$/, '');

  // For prices >= 1, keep at least 2 decimal places
  if (abs >= 1 && dec.length < 2) {
    dec = parts[1].substring(0, 2);
  }

  if (dec.length === 0) return parts[0];
  return `${parts[0]}.${dec}`;
}

export function formatQty(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '-';

  const v = Number(value);
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 1) return v.toFixed(4).replace(/\.?0+$/, '');
  return v.toFixed(8).replace(/\.?0+$/, '');
}

export function formatTf(tf?: number | null): string {
  if (!tf) return '-';
  if (tf >= 1440) return `${tf / 1440}d`;
  if (tf >= 60) return `${tf / 60}h`;
  return `${tf}m`;
}
