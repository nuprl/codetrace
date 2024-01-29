export type Children = string | { [attributeName: string]: string };

export function tag(name: string, selfClosing = false) {
  return (...children: Children[]): string => {
    let props = '';
    const firstChild = children[0];
    if (typeof firstChild === 'object') {
      props = Object.keys(firstChild)
        .map((key) => ` ${key}='${firstChild[key]}'`)
        .join('');
    }

    const content =
      typeof firstChild === 'string' ? children : children.slice(1);

    return selfClosing
      ? `<${name}${props} />`
      : `<${name}${props}>${content.join('')}</${name}>`;
  };
}

export function fragment(...children: Children[]): string {
  return children.join('');
}

export const style = tag('style');
export const title = tag('title');

export const a = tag('a');
export const h1 = tag('h1');
export const h2 = tag('h2');
export const h3 = tag('h3');
export const h4 = tag('h4');
export const h5 = tag('h5');
export const h6 = tag('h6');
export const p = tag('p');
export const b = tag('b');
export const i = tag('i');
export const q = tag('q');
export const strong = tag('strong');
export const em = tag('em');
export const pre = tag('pre');
export const code = tag('code');

export const div = tag('div');
export const span = tag('span');

export const table = tag('table');
export const thead = tag('thead');
export const tbody = tag('tbody');
export const tfoot = tag('tfoot');
export const tr = tag('tr');
export const td = tag('td');
export const th = tag('th');
export const colgroup = tag('colgroup');
export const caption = tag('caption');

export const ul = tag('ul');
export const ol = tag('ol');
export const li = tag('li');
export const dt = tag('dt');
export const dd = tag('dd');

export const details = tag('details');
export const summary = tag('summary');

export const area = tag('area', true);
export const base = tag('base', true);
export const br = tag('br', true);
export const col = tag('col', true);
export const embed = tag('embed', true);
export const hr = tag('hr', true);
export const img = tag('img', true);
export const input = tag('input', true);
export const link = tag('link', true);
export const meta = tag('meta', true);
export const param = tag('param', true);
export const source = tag('source', true);
export const track = tag('track', true);
export const wbr = tag('wbr', true);

export const customTag = (name: <FILL>, ...children: Children[]) => {
  return tag(name)(...children);
};
