export type BookTitle = string;

const booksToChaptersMap: { [name: string]: number } = {
  Genesis: 50,
  Exodus: 40,
  Leviticus: 27,
  Numbers: 36,
  Deuteronomy: 34,
  Joshua: 24,
  Judges: 21,
  Ruth: 4,
  '1 Samuel': 31,
  '2 Samuel': 24,
  '1 Kings': 22,
  '2 Kings': 25,
  '1 Chronicles': 29,
  '2 Chronicles': 36,
  Ezra: 10,
  Nehemiah: 13,
  Esther: 10,
  Job: 42,
  Psalms: 150,
  Proverbs: 31,
  Ecclesiastes: 12,
  'Song of Solomon': 8,
  Isaiah: 66,
  Jeremiah: 52,
  Lamentations: 5,
  Ezekial: 48,
  Daniel: 12,
  Hosea: 14,
  Joel: 3,
  Amos: 9,
  Obadiah: 1,
  Jonah: 4,
  Micah: 7,
  Nahum: 3,
  Habakkuk: 3,
  Zephaniah: 3,
  Haggai: 2,
  Zechariah: 14,
  Malachi: 4,
  Matthew: 28,
  Mark: 16,
  Luke: 24,
  John: 21,
  Acts: 28,
  Romans: 16,
  '1 Corinthians': 16,
  '2 Corinthians': 13,
  Galatians: 6,
  Ephesians: 6,
  Philippians: 4,
  Colossians: 4,
  '1 Thessalonians': 5,
  '2 Thessalonians': 3,
  '1 Timothy': 6,
  '2 Timothy': 4,
  Titus: 3,
  Philemon: 1,
  Hebrews: 13,
  James: 5,
  '1 Peter': 5,
  '2 Peter': 3,
  '1 John': 5,
  '2 John': 1,
  '3 John': 1,
  Jude: 1,
  Revelation: 22,
};

const possibleTitles: string[] = Array.from(Object.keys(booksToChaptersMap));

let i = 0;
export const hebrewScriptures: BookInfo[] = possibleTitles
  .slice(0, 39)
  .map((x) => {
    i++;
    return { id: i, title: x, chapters: booksToChaptersMap[x] };
  });

export const newTestament: BookInfo[] = possibleTitles.slice(39).map((x) => {
  i++;
  return { id: i, title: x, chapters: booksToChaptersMap[x] };
});

export const getBookTitleFromSlug = (
  bookTitle: BookTitle | unknown,
): BookTitle => {
  if (typeof bookTitle !== 'string') {
    throw new Error(`Booktitle ${bookTitle} is not even string`);
  }

  const found = possibleTitles.find(
    (x) => bookTitle === x.replace(' ', '-').toLowerCase(),
  );

  if (found === undefined) {
    throw new Error('There is not such book title:${bookParam}');
  }

  return found;
};

export const getChapterFromSlug = (
  chapter: number | unknown,
  book: BookTitle,
): number => {
  if (isNaN(Number(chapter))) {
    throw new Error(`Chapter "${chapter}" is not a number`);
  }
  if (!Number.isInteger(Number(chapter))) {
    throw new Error(`Chapter "${chapter}" is not integer`);
  }
  if (Number(chapter) > booksToChaptersMap[book]) {
    throw new Error(`Chapter "${chapter}" is too hight`);
  }
  if (Number(chapter) < 1) {
    throw new Error(`Chapter "${chapter}" is too low`);
  }
  return Number(chapter);
};

export interface BookInfo {
  id: number;
  title: string;
  chapters: number;
}
