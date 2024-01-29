export interface TimeDelta {
  sign: '' | '+',
  years: number,
  months: number,
  days: number,
  hours: <FILL>,
  minutes: number,
  seconds: number,
}

export interface AgeOptions {
  /**
   * Comparison date
   */
  now?: Date;

  /**
   * Precision
   */
  levels?: number;
}

/**
 * Get the ymdhms differences between two dates
 *
 * @param left  + date value
 * @param right - date value
 * @returns     difference between the two dates
 */
export function getTimeDelta(
  left: Date,
  right: Date,
): TimeDelta {
  const {
    secondsPerSecond: secondsPerSecond,
    secondsPerMinute: secondsPerMinute,
    secondsPerHour: secondsPerHour,
    secondsPerDay: secondsPerDay,
    secondsPerMonth: secondsPerMonth,
    secondsPerYear: secondsPerYear,
  } = getAge.defaults;

  const leftMs = left.valueOf();
  const rightMs = right.valueOf();
  let delta = Math.abs(leftMs - rightMs) / 1000;

  const years = Math.floor(delta / secondsPerYear);
  delta -= years * secondsPerYear;

  const months = Math.floor(delta / secondsPerMonth);
  delta -= months * secondsPerMonth;

  const days = Math.floor(delta / secondsPerDay);
  delta -= days * secondsPerDay;

  const hours = Math.floor(delta / secondsPerHour);
  delta -= hours * secondsPerHour;

  const minutes = Math.floor(delta / secondsPerMinute);
  delta -= minutes * secondsPerMinute;

  const seconds = Math.floor(delta / secondsPerSecond);
  delta -= seconds * secondsPerSecond;

  const timeDelta: TimeDelta = {
    sign: leftMs < rightMs ? '+' : '',
    years,
    months,
    days,
    hours,
    minutes,
    seconds,
  };

  return timeDelta;
}


/**
 * Get the approximate age of the given date from now
 *
 * @param birth     date to find the age of
 * @param options
 * @returns         approximate age of the date
 *
 * @example
 * ```ts
 * '2y3m4d'
 * '2y4d1h'
 * '3y2m1s'
 * '17s'
 * ```
 */
export function getAge(
  birth: Date,
  options?: AgeOptions
): string {
  const now = options?.now ?? getAge.defaults.NOW();
  const levels = options?.levels ?? getAge.defaults.LEVELS;
  const delta = getTimeDelta(now, birth);
  let str = delta.sign;
  let cnt = 0;

  if (levels < 0) {
    throw new TypeError(`getAge(): "levels" must be >= 0. Given: ${levels}.`);
  }

  // years

  if (delta.years) {
    str += `${delta.years}y`;
    cnt += 1;
  }

  // months

  if (cnt >= levels) return str;

  if (delta.months) {
    str += `${delta.months}m`;
    cnt += 1;
  }

  // days

  if (cnt >= levels) return str;

  if (delta.days) {
    str += `${delta.days}d`;
    cnt += 1;
  }

  // hours

  if (cnt >= levels) return str;

  if (delta.hours) {
    str += `${delta.hours}h`;
    cnt += 1;
  }

  // minutes

  if (cnt >= levels) return str;

  if (delta.minutes) {
    str += `${delta.minutes}m`;
    cnt += 1;
  }

  // seconds

  if (cnt >= levels) return str;

  if (delta.seconds) {
    str += `${delta.seconds}s`;
    cnt += 1;
  }

  // if there is no difference, show zero seconds...

  if (cnt === 0) {
    str += '0s';
  }

  return str;
}

/**
 * defaults for usage with `@nkp/age` functions
 */
getAge.defaults = {
  secondsPerSecond: 1,
  secondsPerMinute: 1 * 60,
  secondsPerHour  : 1 * 60 * 60,
  secondsPerDay   : 1 * 60 * 60 * 24,
  // on average 30.437 days in a month
  secondsPerMonth : 1 * 60 * 60 * 24 * 30.437,
  // SECONDS_PER_MONTH : 1 * 60 * 60 * 24 * 30,
  // on average 365.24 days in a year
  secondsPerYear  : 1 * 60 * 60 * 24 * 365.24,
  // SECONDS_PER_YEAR  : 1 * 60 * 60 * 24 * 365,

  NOW: (): Date => new Date(),
  LEVELS: 3,
};
