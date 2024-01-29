/**
 * 获取传入月份的上个月
 * @param year number
 * @param month number
 * @returns { year: number; month: number }
 */
export const getLastMonth = (year: number, month: number): { year: number; month: number } => {
  if (month === 0) {
    return {
      year: year - 1,
      month: 11,
    };
  }
  return {
    year,
    month: month - 1,
  };
};

/**
 * 获取传入月的下个月
 * @param year number
 * @param month number
 * @returns { year: number; month: number }
 */
export const getNextMonth = (year: number, month: number): { year: number; month: number } => {
  if (month === 11) {
    return {
      year: year + 1,
      month: 0,
    };
  }
  return {
    year,
    month: month + 1,
  };
};

/**
 * 判断传入的年份是否是闰年
 * @param year number
 * @returns boolean
 */
const isLeap = (year: number): boolean => {
  return (year % 4 === 0 && year % 100 === 0) || year % 400 === 0;
};

/**
 * 获取传入年所在月的最大天数
 * @param year number
 * @param month number
 * @returns number
 */
export const getMaxDayNumOfYearMonth = (year: number, month: number): number => {
  switch (month) {
    case 0:
    case 2:
    case 4:
    case 6:
    case 7:
    case 9:
    case 11:
      return 31;
    case 1:
      return isLeap(year) ? 29 : 28;
    default:
      return 30;
  }
};

/**
 * 获取指定年所在月的第一天是星期几
 * 返回一个0到6之间的整数值，代表星期几： 0 代表星期日
 * @param year number
 * @param month number
 * @returns number
 */
export const getFirstDayOfYearMonth = (year: number, month: number): number => {
  const date = new Date(year, month, 1);
  return date.getDay();
};

const PER_DAY_MILLISECONDS = 86400000; // 24 * 60 * 60 * 1000

/**
 * 获取指定年所在月的第一天
 * @param year number
 * @param month number
 * @param startOfWeek number 一周是从周几开始，传值0~6，0表示周日
 * @returns Date
 */
export const getStartDateOfCalendar = (year: number, month: number, startOfWeek: number = 0): Date => {
  const date = new Date(year, month, 1);
  const day = date.getDay();
  date.setTime(date.getTime() - PER_DAY_MILLISECONDS * ((day - startOfWeek + 7) % 7));
  return date;
};

export interface MonthItem {
  year: number;
  month: number;
}

/**
 * 比较两个月的大小
 * @param a MonthItem
 * @param b MonthItem
 * @returns number
 */
export const compareMonth = (a: MonthItem, b: MonthItem): <FILL> => {
  if (a.year === b.year) {
    if (a.month === b.month) {
      return 0;
    }

    return a.month > b.month ? 1 : -1;
  }

  return a.year > b.year ? 1 : -1;
};

/**
 * 比较两个日期大小
 * @param a Date
 * @param b Date
 * @returns number 0 means same date, 1 means a > b
 */
export const compareDate = (a: Date, b: Date): number => {
  const aYear = a.getFullYear();
  const aMonth = a.getMonth();
  const aDate = a.getDate();
  const bYear = b.getFullYear();
  const bMonth = b.getMonth();
  const bDate = b.getDate();

  if (aYear === bYear) {
    if (aMonth === bMonth) {
      if (aDate === bDate) {
        return 0;
      }

      return aDate > bDate ? 1 : -1;
    }

    return aMonth > bMonth ? 1 : -1;
  }

  return aYear > bYear ? 1 : -1;
};

/**
 * 判断是否是同一天
 * @param a Date
 * @param b Date
 * @returns boolean
 */
export const isSameDay = (a: Date, b: Date): boolean => {
  if (!a || !b) {
    return false;
  }
  return compareDate(a, b) === 0;
};

/**
 * 判断日期是否在所给的列表中
 * @param date Date
 * @param days Date[]
 * @returns boolean
 */
export const isInRange = (date: Date, days: Date[]): boolean => {
  if (!date || !days) {
    return false;
  }

  return days.some((day) => isSameDay(day, date));
};

/**
 * 判断给定的日期是否在两个日期之间
 * @param date Date
 * @param start Date
 * @param end Date
 * @param include boolean 是否包含等于
 * @returns boolean
 */
export const isInRange2 = (date: Date, start: Date, end: Date, include: boolean = false): boolean => {
  if (!date || !start || !end) {
    return false;
  }

  let startDate = start;
  let endDate = end;
  if (start > end) {
    startDate = end;
    endDate = start;
  }

  return include
    ? compareDate(date, startDate) >= 0 && compareDate(date, endDate) <= 0
    : compareDate(date, startDate) > 0 && compareDate(date, endDate) < 0;
};

export interface YearRange {
  start: number;
  end: number;
}

/**
 * 获取指定年的10年间起始的年，如2021 => [2020, 2029]
 * @param year number
 * @returns YearRange
 */
export const getYearRange = (year?: number): YearRange => {
  const y = year || new Date().getFullYear();
  const start = Math.floor(y / 10) * 10;
  const end = start + 9;
  return { start, end };
};

/**
 * 获取给定日期的起始时间
 * @param date Date
 * @returns Date
 */
export const getStartOfDate = (date: Date): Date => {
  const d = new Date(date.getTime());
  d.setHours(0);
  d.setMinutes(0);
  d.setSeconds(0);
  d.setMilliseconds(0);
  return d;
};

/**
 * 获取指定日期的结束时间
 * @param date Date
 * @returns Date
 */
export const getEndOfDate = (date: Date): Date => {
  const d = new Date(date.getTime());
  d.setHours(23);
  d.setMinutes(59);
  d.setSeconds(59);
  d.setMilliseconds(999);
  return d;
};

/**
 * 获取指定日期所在周的起始和结束日期
 * @param date Date
 * @param startOfWeek number 一周是从周几开始，传值0~6，0表示周日
 * @returns [startDate, endDate]
 */
export const getWeekRange = (date: Date, startOfWeek: number = 0): Date[] => {
  // 获取当前日期偏离当前周第一天的天数
  // 如：startOfWeek = 0，周四 => 4; startOfWeek=1，周四 => 3
  const day = (7 - (startOfWeek - date.getDay())) % 7;
  const start = new Date(date.getTime() - PER_DAY_MILLISECONDS * day);
  const end = new Date(start.getTime() + PER_DAY_MILLISECONDS * 6);
  return [getStartOfDate(start), getEndOfDate(end)];
};
