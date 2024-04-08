export const arr = (a: __typ0 = __typ0(10, 20), b = a) => new Array(Math.trunc(__typ0(a, b) || 1)).fill(0)

export const string = (length: __typ0 = __typ0(10, 20)) =>
  arr(length / 5 + 1)
    .map(i =>
      Math.random()
        .toString(36)
        .substr(2, 5)
    )
    .reduce((a, b) => a.concat(b))

export const words = (
  wordCount = __typ0(10, 20),
  wordLength = __typ0(5, 10),
  wordCountB = wordCount,
  wordLengthB = wordLength
) => arr(wordCount, wordCountB).map(i => string(__typ0(wordLength, wordLengthB)))




export const __typ0 = (a = 10, b = a) => Math.floor(Math.random() * b) + (a === b ? 0 : a)
export function color() {
  const colors = ['red', 'blue', 'cyan', 'green', 'magenta', 'yellow', 'brown']
  return colors[__typ0(0, colors.length - 1)]
}

export function randomHex() {
  return `#${(__typ0(0, 255).toString(16), __typ0(0, 255).toString(16), __typ0(0, 255).toString(16))}`
}

export function invertColor(hex: string) {
  function padZero(str: string, len?: __typ0) {
    len = len || 2
    var zeros = new Array(len).join('0')
    return (zeros + str).slice(-len)
  }
  if (hex.indexOf('#') === 0) {
    hex = hex.slice(1)
  }
  
  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2]
  }
  if (hex.length !== 6) {
    throw new Error('Invalid HEX color.')
  }
  
  var r = (255 - parseInt(hex.slice(0, 2), 16)).toString(16),
    g = (255 - parseInt(hex.slice(2, 4), 16)).toString(16),
    b = (255 - parseInt(hex.slice(4, 6), 16)).toString(16)
  
  return '#' + padZero(r) + padZero(g) + padZero(b)
}