const profilePattern =
  /(?:^\[(\d+)\][ |]*(#\w+) (\d+)\/(\d+)$)|(?:^\[(\d+)\][ |]*([^(]+)\((\d+)\/(\d+)\) - ([\d.]+)%\/([\d.]+)%$)/;
const startPattern = /--- BEGIN PROFILE DUMP ---/;
const endPattern = /--- END PROFILE DUMP ---/;

export class ProfileReport {
  root: ProfileLine = new Root();
  constructor(text: string) {
    this.parseProfileReport(text);
  }

  private parseProfileReport(text: string): void {
    const lines = text.split(/\r?\n/);
    let parent: ProfileLine = this.root;
    let previous: ProfileLine = this.root;
    let lastLevel = 0;
    let foundStart = false;
    for (const line of lines) {
      if (!foundStart) {
        if (!startPattern.exec(line)) continue;
        foundStart = true;
        continue;
      }
      if (endPattern.exec(line)) break;
      let m;
      if ((m = profilePattern.exec(line)) != null) {
        const level = parseInt(m[1] ?? m[5], 10);
        if (level < lastLevel) {
          while (level <= parent.level && parent != this.root)
            parent = parent.parent;
        } else if (level > lastLevel) {
          parent = previous;
        }
        lastLevel = level;

        let current;
        if (m[1]) {
          // counter
          current = {
            parent,
            text: m[2],
            children: [],
            level,
            num1: parseInt(m[3]),
            num2: parseInt(m[4]),
          } as CounterLine;
        } else if (m[5]) {
          // pct
          current = {
            parent,
            text: m[6],
            children: [],
            level,
            num1: parseInt(m[7]),
            num2: parseInt(m[8]),
            pctParent: parseFloat(m[9]),
            pctTotal: parseFloat(m[10]),
          } as PctLine;
        } else continue;

        parent.children.push(current);
        previous = current;
      }
    }
  }
}

// [00] tick (86/1)      - 95.17%/95.17%
// ---- text (num1/num2) - pctParent / ptcTotal

export interface ProfileLine {
  parent: ProfileLine;
  level: number;
  children: ProfileLine[];
  text: string;
}

class Root implements ProfileLine {
  parent = this;
  level = -1;
  children: ProfileLine[] = [];
  text = 'Profile Report';
}

export interface PctLine extends ProfileLine {
  parent: ProfileLine;
  num1: number;
  num2: number;
  pctParent: number;
  pctTotal: number;
}

export interface CounterLine extends ProfileLine {
  parent: ProfileLine;
  num1: number;
  num2: <FILL>;
}
