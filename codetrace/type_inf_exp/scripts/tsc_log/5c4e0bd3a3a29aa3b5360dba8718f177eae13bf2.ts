type __typ0 = any;

interface __typ2 {
    onParseIntegers?(interger: __typ1): __typ0;
    onParseErrors?(error: string): __typ0;
    onParseStrings?(data: string): __typ0;
    onParseBulkSrings?(bulkString: string): __typ0;
    onParseArray?(dataArr: string[]): __typ0;
}

export default class RedisParser {
    options: __typ2
    constructor(options: __typ2) {
        this.options = options
    }
    public parse(data: string) {
        const lines = data.split("\r\n")
        while (lines.length) {
            const line = lines[0]
            const char = line[0];
            switch (char) {
                case ":": {
                    const interger = this.parseIntegers(line)
                    lines.shift();
                    this.options.onParseIntegers && this.options.onParseIntegers(interger)
                    break;
                }
                case "+": {
                    const data = this.parseStrings(line);
                    lines.shift();
                    this.options.onParseStrings && this.options.onParseStrings(data)
                    break
                }
                case "-": {
                    const message = this.parseErrors(line)
                    lines.shift();
                    this.options.onParseErrors && this.options.onParseErrors(message);
                    break
                }
                case "$": {
                    const bulkString = this.parseBulkString(lines);
                    this.options.onParseBulkSrings && this.options.onParseBulkSrings(bulkString);
                    break
                }
                case "*": {
                    const dataArr = this.parseArray(lines);
                    this.options.onParseArray && this.options.onParseArray(dataArr)
                    break
                }
                default: {
                    lines.shift()
                }
            }
        }
    }
    private parseArray(lines: string[]): __typ0[] {
        const numberData = lines.shift() as string;
        let __typ1 = +String.prototype.slice.call(numberData, 1);
        const dataArr = [];
        while (__typ1 > 0) {
            const line = lines[0]
            const char = line[0];
            switch (char) {
                case ":": {
                    dataArr.push(this.parseIntegers(line));
                    lines.shift();
                    break;
                }
                case "+": {
                    dataArr.push(this.parseStrings(line));
                    lines.shift();
                    break
                }
                case "$": {
                    dataArr.push(this.parseBulkString(lines))
                    break
                }
            }
            __typ1--;
        }
        return dataArr
    }
    private parseBulkString(lines: string[]): string {
        const numberData = lines.shift() as string;
        const __typ1 = String.prototype.slice.call(numberData, 1);
        if (+__typ1 >= 0) {
            const bulkString = lines.shift() as string;
            return bulkString
        } else {
            return "" 
             
        }
    }
    private parseIntegers(line: string): __typ1 {
        return +String.prototype.slice.call(line, 1)
    }
    private parseErrors(line: string): string {
        return String.prototype.slice.call(line, 5)
    }
    private parseStrings(line: string): string {
        return String.prototype.slice.call(line, 1)
    }
}