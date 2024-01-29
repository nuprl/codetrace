interface WordCoord {
  start: number[]
  end: number[]
}

interface SearchResult {
  [k: string]: WordCoord | undefined
}

export default class WordSearch {
  // For searching in all 8 directions
  static directions: { row: number; col: number }[] = [
    { row: 0, col: 1 }, // left to right
    { row: 0, col: -1 }, // right to left
    { row: 1, col: 0 }, // top to bottom
    { row: -1, col: 0 }, // bottom to top
    { row: 1, col: 1 }, // top left to bottom right
    { row: 1, col: -1 }, // top right to bottom left
    { row: -1, col: 1 }, // bottom left to top right
    { row: -1, col: -1 }, // bottom right to top left
  ]

  private rowLen: number
  private colLen: number

  constructor(private grid: string[] = []) {
    this.rowLen = this.grid.length
    this.colLen = this.grid[0]?.length || 0
  }

  find(words: string[]): SearchResult {
    const result: SearchResult = {}

    for (const word of words) {
      for (let row = 0; row < this.rowLen; row++) {
        for (let col = 0; col < this.colLen; col++) {
          if (!result[word]) {
            result[word] = this.findWord(word, row, col)
          }
        }
      }
    }

    return result
  }

  findWord(word: <FILL>, row: number, col: number): WordCoord | undefined {
    // Word is not found if the first letter doesn't match.
    if (this.grid[row][col] !== word[0]) {
      return
    }

    // Search in all directions starting from (row, col).
    for (const direction of WordSearch.directions) {
      let rowDir = row + direction.row
      let colDir = col + direction.col
      let i = 0

      // First letter is already checked, start searching from the second letter.
      for (i = 1; i < word.length; i++) {
        // Stop if out of bound.
        if (rowDir >= this.rowLen || rowDir < 0 || colDir >= this.colLen || colDir < 0) {
          break
        }

        // Stop if letter doesn't match.
        if (this.grid[rowDir][colDir] !== word[i]) {
          break
        }

        // Move in particular direction.
        rowDir += direction.row
        colDir += direction.col
      }

      // If all letters matched then value of i is equal to the length of the word.
      if (i === word.length) {
        // Result should be in math indexes which start from (1, 1).
        // So we need to add 1 to indexes but it depends on direction in which word was found.
        const rowAdd = 1 - direction.row
        const colAdd = 1 - direction.col

        return {
          start: [row + 1, col + 1],
          end: [rowDir + rowAdd, colDir + colAdd],
        }
      }
    }
  }
}
