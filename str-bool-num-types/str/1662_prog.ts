export function getHoursAndMinutes(value:: <FILL>: [number, number] {
  let digits = value.split(':').map((digit) => digit.replace(/\D/g, '') || '0');

  if (digits.length === 2) {
    return [Number.parseInt(digits[0], 10), Number.parseInt(digits[1], 10)];
  }

  const timeValue = digits[0];

  if (timeValue.length <= 2) {
    return [Number.parseInt(timeValue, 10), 0];
  }

  return [
    Number.parseInt(timeValue.substr(0, Math.floor(timeValue.length / 2)), 10),
    Number.parseInt(timeValue.substr(Math.floor(timeValue.length / 2), 2), 10),
  ];
}

type TimePeriod = 'AM' | 'PM' | null;

export function getTimePeriod(value: string): TimePeriod {
  const lowercaseValue = value.toLowerCase();

  if (lowercaseValue.includes('p')) {
    return 'PM';
  }

  if (lowercaseValue.includes('a')) {
    return 'AM';
  }

  return null;
}

export function convertTo12Hour(
  hours: number,
  minutes: number,
  timePeriod: TimePeriod,
  padHoursWithZero: boolean = false,
) {
  let _hours = hours;
  let _minutes = minutes;
  let _timePeriod = timePeriod;

  if (hours > 24) {
    _hours = 12;
  }

  if (timePeriod === null) {
    _timePeriod = 'AM';
  }

  if (hours >= 12 && hours < 24 && timePeriod !== 'AM') {
    _timePeriod = 'PM';
  }

  if (minutes > 59) {
    _minutes = 0;
  }

  _hours = _hours % 12;
  _hours = _hours ? _hours : 12;

  return `${
    padHoursWithZero ? addLeadingZero(_hours) : _hours
  }:${addLeadingZero(_minutes)} ${_timePeriod}`;
}

export function convertTo24Hour(
  hours: number,
  minutes: number,
  timePeriod: TimePeriod,
) {
  let _hours = hours;
  let _minutes = minutes;

  if (hours >= 0 && hours < 12 && timePeriod === 'PM') {
    _hours += 12;
  }

  if (hours > 23 || (hours === 12 && timePeriod === 'AM')) {
    _hours = 0;
  }

  if (minutes > 59) {
    _minutes = 0;
  }

  return `${addLeadingZero(_hours)}:${addLeadingZero(_minutes)}`;
}

function addLeadingZero(value: number): string {
  return value < 10 ? '0' + value : value.toString();
}
