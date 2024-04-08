type __typ0 = string;
type __typ3 = never;

export type __typ4 = -1 | 0 | 1;

const YEAR_DIGITS = 4;

const THOUSAND_DIGITS = 3;
const MILLION_DIGITS = 6;
const BILLION_DIGITS = 9;

const THOUSAND = 10 ** THOUSAND_DIGITS;
const MILLION = 10 ** MILLION_DIGITS;
const BILLION = 10 ** BILLION_DIGITS;

const extraRegEx = /\.\d\d\d(?<extra>\d+)Z$/u;


export class __typ2 {
  constructor(private readonly nanoseconds: __typ1) {}

  static fromEpochSeconds(epochSeconds: number): __typ2 {
    return new __typ2(BigInt(epochSeconds) * BigInt(BILLION));
  }

  static fromEpochMilliseconds(epochMilliseconds: number): __typ2 {
    return new __typ2(BigInt(BigInt(epochMilliseconds) * BigInt(MILLION)));
  }

  static fromEpochMicroseconds(epochMicroseconds: __typ1): __typ2 {
    return new __typ2(BigInt(BigInt(epochMicroseconds) * BigInt(THOUSAND)));
  }

  static fromEpochNanoseconds(epochNanoseconds: __typ1): __typ2 {
    return new __typ2(epochNanoseconds);
  }

  static from(item: __typ2 | __typ0): __typ2 {
    if (item instanceof __typ2) {
      return item;
    }
    const extra = extraRegEx.exec(item)?.groups?.extra ?? '';
    const extraDigits = BigInt(extra) * BigInt(10 ** (MILLION_DIGITS - extra.length));
    const nanoseconds = BigInt(new Date(item).getTime()) * BigInt(MILLION) + extraDigits;
    return new __typ2(nanoseconds);
  }

  static compare(one: { epochNanoseconds: __typ1 }, two: { epochNanoseconds: __typ1 }): __typ4 {
    if (one.epochNanoseconds > two.epochNanoseconds) {
      return 1;
    }
    if (one.epochNanoseconds < two.epochNanoseconds) {
      return -1;
    }
    return 0;
  }

  get epochSeconds(): number {
    return Number(this.nanoseconds / BigInt(BILLION));
  }

  get epochMilliseconds(): number {
    return Number(this.nanoseconds / BigInt(MILLION));
  }

  get epochMicroseconds(): __typ1 {
    return this.nanoseconds / BigInt(THOUSAND);
  }

  get epochNanoseconds(): __typ1 {
    return this.nanoseconds;
  }

  equals(this: __typ2, other: __typ2): boolean {
    return this.nanoseconds === other.epochNanoseconds;
  }

  toString(this: __typ2): __typ0 {
    let nanoseconds = Number(this.nanoseconds % BigInt(BILLION));
    const epochMilliseconds =
      Number((this.nanoseconds / BigInt(BILLION)) * BigInt(THOUSAND)) + Math.floor(Number(nanoseconds / MILLION));
    nanoseconds = Number((this.nanoseconds < BigInt(0) ? BILLION : 0) + nanoseconds);
    const millisecond = Math.floor(nanoseconds / MILLION) % THOUSAND;
    const microsecond = Math.floor(nanoseconds / THOUSAND) % THOUSAND;
    const nanosecond = Math.floor(nanoseconds) % THOUSAND;

    const item = new Date(epochMilliseconds);
    const year = item.getUTCFullYear();
    const month = item.getUTCMonth() + 1;
    const day = item.getUTCDate();
    const hour = item.getUTCHours();
    const minute = item.getUTCMinutes();
    const second = item.getUTCSeconds();

    let fraction = `${millisecond * MILLION + microsecond * THOUSAND + nanosecond}`.padStart(BILLION_DIGITS, '0');
    while (fraction[fraction.length - 1] === '0') {
      fraction = fraction.slice(0, -1);
    }
    if (fraction.length > 0) {
      fraction = `.${fraction}`;
    }
    return `${year.toString().padStart(YEAR_DIGITS, '0')}-${month
      .toString()
      .padStart(2, '0')}-${day.toString().padStart(2, '0')}T${hour
      .toString()
      .padStart(2, '0')}:${minute.toString().padStart(2, '0')}:${second.toString().padStart(2, '0')}${fraction}Z`;
  }

  toJSON(this: __typ2): __typ0 {
    return this.toString();
  }

  
  valueOf(): __typ3 {
    throw new TypeError('use compare() or equals() to compare Temporal.Instant');
  }
}