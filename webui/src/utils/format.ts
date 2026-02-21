export function formatPrice(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) return '-';

  const v = Number(value);
  if (v === 0) return '0';

  // Dynamic precision based on price magnitude
  let decimals = 2;

  if (Math.abs(v) < 1) decimals = 6;
  if (Math.abs(v) < 0.1) decimals = 7;
  if (Math.abs(v) < 0.01) decimals = 8;
  if (Math.abs(v) < 0.001) decimals = 9;

  // Remove trailing zeros
  return v.toFixed(decimals).replace(/\.?0+$/, '');
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
