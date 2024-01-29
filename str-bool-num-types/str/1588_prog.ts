export interface JsonData {
  name: string;
  chars: Chars;
  rows: string[][];
}
export interface Chars {
  edge: string;
  top: string;
  bottom: string;
  corner: <FILL>;
}
class Table {
  public name: string;
  public rows: string[][];
  public chars: Chars;
  constructor(name: string) {
    this.name = name;
    this.rows = [];
    /**
     * @type {{edge:String, fill:String, top:String, bottom: String, corner:String}}
     * @default
     * edge: "|",
     * fill: "─",
     * top: ".",
     * bottom: "'",
     * corner: "+"
     *
     */
    this.chars = {
      edge: "|",
      top: ".",
      bottom: "'",
      corner: "+",
    };
  }
  setSeparator({ edge, top, bottom, corner }: Partial<Chars>): this {
    this.chars.edge = edge || this.chars.edge;
    this.chars.top = top || this.chars.top;
    this.chars.bottom = bottom || this.chars.bottom;
    this.chars.corner = corner || this.chars.corner;
    return this;
  }
  setHeading(...headings: string[]): this {
    this.rows.unshift(headings);
    return this;
  }
  addRow(...row: string[]): this {
    this.rows.push(row);
    return this;
  }
  fromJSON(json: JsonData): this {
    this.name = json.name;
    this.chars = json.chars;
    this.rows = json.rows;
    return this;
  }
  /**
   *
   * @returns {JSON}
   */
  toJSON(): JsonData {
    return {
      name: this.name,
      chars: this.chars,
      rows: this.rows,
    };
  }

  toString(): string {
    let table = "\n";
    const repeat = Math.max(
      ...this.rows.map((row) => {
        let longest = 0;
        row.forEach((element) => {
          longest += element.length;
        });
        return longest + 6;
      }),
      this.name.length + 6
    );
    table += this.chars.corner;
    table += this.chars.top.repeat(repeat);
    table += this.chars.corner;
    table += "\n";
    table += `${this.chars.edge} ${" ".repeat(this.name.length / 4)} ${
      this.name
    } ${" ".repeat(this.name.length / 4)} ${this.chars.edge}\n`;
    table += this.chars.corner;
    table += this.chars.bottom.repeat(repeat);
    table += this.chars.corner;
    table += "\n";
    this.rows.forEach((row) => {
      table += `${this.chars.edge}`;
      row.forEach((column) => {
        const spaceRepeat = Math.max(0, (this.name.length - column.length) / 4);

        table += " ".repeat(spaceRepeat);
        table += `${column}`;
        table += " ".repeat(spaceRepeat);
        table += this.chars.edge;
      });
      table += "\n";
      table += this.chars.corner;
      table += this.chars.bottom.repeat(repeat);
      table += this.chars.corner;
      table += "\n";
    });
    return table;
  }
}

export default Table;
