/**!
 * @see https://github.com/mdevils/html-entities
 * @license MIT
 * @author mdevils "Marat Dulin"
 */

type StringMap = { [x: string]: string };
/**
 * @see https://github.com/mdevils/html-entities/blob/master/src/named-references.ts
 */
export const namedXmlCharacters: StringMap = {
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&apos;",
  "&": "&amp;",
};
export const namedXmlEntities: StringMap = {};
Object.keys(namedXmlCharacters).forEach((ch) => (namedXmlEntities[namedXmlCharacters[ch]] = ch));

const fromCharCode = String.fromCharCode;
const fromCodePoint =
  String.fromCodePoint ||
  function (astralCodePoint: number) {
    return String.fromCharCode(
      Math.floor((astralCodePoint - 0x10000) / 0x400) + 0xd800,
      ((astralCodePoint - 0x10000) % 0x400) + 0xdc00
    );
  };
const outOfBoundsChar = fromCharCode(65533);
const numericUnicodeMap: Record<number, number> = {
  0: 65533,
  128: 8364,
  130: 8218,
  131: 402,
  132: 8222,
  133: 8230,
  134: 8224,
  135: 8225,
  136: 710,
  137: 8240,
  138: 352,
  139: 8249,
  140: 338,
  142: 381,
  145: 8216,
  146: 8217,
  147: 8220,
  148: 8221,
  149: 8226,
  150: 8211,
  151: 8212,
  152: 732,
  153: 8482,
  154: 353,
  155: 8250,
  156: 339,
  158: 382,
  159: 376,
};

/**
 * @see https://github.com/mdevils/html-entities/blob/master/src/index.ts
 */
export function decodeXmlEntity(entity: string): string {
  if (!entity) return "";

  let decodeResult = entity;
  const decodeResultByReference = namedXmlEntities[entity];
  if (decodeResultByReference) {
    decodeResult = decodeResultByReference;
  } else if (entity[0] === "&" && entity[1] === "#") {
    const decodeSecondChar = entity[2];
    const decodeCode =
      decodeSecondChar == "x" || decodeSecondChar == "X"
        ? parseInt(entity.substr(3), 16)
        : parseInt(entity.substr(2));

    decodeResult =
      decodeCode >= 0x10ffff
        ? outOfBoundsChar
        : decodeCode > 65535
        ? fromCodePoint(decodeCode)
        : fromCharCode(numericUnicodeMap[decodeCode] || decodeCode);
  }
  return decodeResult;
}
export function decodeXml(xml: <FILL>): string {
  if (!xml) return "";

  const macroRegExp = /&(?:#\d+|#[xX][\da-fA-F]+|[0-9a-zA-Z]+);/g;
  macroRegExp.lastIndex = 0;
  let replaceMatch = macroRegExp.exec(xml);
  let replaceResult: string;
  if (replaceMatch) {
    replaceResult = "";
    let replaceLastIndex = 0;
    do {
      if (replaceLastIndex !== replaceMatch.index) {
        replaceResult += xml.substring(replaceLastIndex, replaceMatch.index);
      }
      const replaceInput = replaceMatch[0];
      replaceResult += decodeXmlEntity(replaceInput);
      replaceLastIndex = replaceMatch.index + replaceInput.length;
    } while ((replaceMatch = macroRegExp.exec(xml)));

    if (replaceLastIndex !== xml.length) {
      replaceResult += xml.substring(replaceLastIndex);
    }
  } else {
    replaceResult = xml;
  }
  return replaceResult;
}
