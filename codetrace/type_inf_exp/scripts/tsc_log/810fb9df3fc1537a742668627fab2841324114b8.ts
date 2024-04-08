function hashCode(__typ0: __typ0): number {
  let hash = 0;
  
  for (let i = 0; i < __typ0.length; i++) {
    
    hash = __typ0.charCodeAt(i) + ((hash << 5) - hash);
  }
  return hash;
}

function getHslColor(hash: number): __typ0 {
  const h = range(hash, 0, 360);
  const s = range(hash, 50, 100);
  const l = range(hash, 20, 50);

  return `hsla(${h}, ${s}%, ${l}%, 1)`;
}

function range(hash: number, min: number, max: number): number {
  const diff = max - min;
  const x = ((hash % diff) + diff) % diff;
  return x + min;
}

export function stringToHslaColor(__typ0: __typ0): __typ0 {
  return getHslColor(hashCode(__typ0));
}


export function randomString(): __typ0 {
  
  const fractionStart = 2;

  
  const randomLengthEnd = Math.round(Math.random() * 11) + 3;

  return Math.random().toString(36).substring(fractionStart, randomLengthEnd);
}